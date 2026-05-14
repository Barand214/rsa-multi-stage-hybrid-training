from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric
from torch_geometric.data import Data
from python_scripts.Project_config import device
from torch.distributions import Normal, Categorical, Categorical


# ===================== 傅里叶动作空间 =====================
class FourierActionSpace:
    """
    傅里叶参数动作空间包装器
    """
    def __init__(self, n_servos=20, max_harmonics=20, T_max=10.0):
        self.n_servos = n_servos
        self.max_harmonics = max_harmonics
        self.T_max = T_max
        
        # 参数维度说明：
        # 1. n_harmonics: 1个离散参数 [1, max_harmonics]
        # 2. T: 1个连续参数 (0, T_max]
        # 3. A: n_servos * max_harmonics个参数 [-1, 1]
        # 4. ω: max_harmonics个参数 (0, ∞)
        # 5. φ: n_servos * max_harmonics个参数 [0, T]
        
        # 总连续参数维度
        self.continuous_param_dim = 1 + (n_servos * max_harmonics) + max_harmonics + (n_servos * max_harmonics)
        
    def sample_random(self):
        """随机采样参数（用于初始化）"""
        # n_harmonics: 离散值 [1, max_harmonics]
        n = np.random.randint(1, self.max_harmonics + 1)
        
        # T: 连续值 (0, T_max]
        T = np.random.uniform(0.1, self.T_max)
        
        # A_i ∈ [-1, 1]
        A = np.random.uniform(-1, 1, (self.n_servos, self.max_harmonics))
        
        # ω_i ∈ (0, ∞)，实际用(0, 20π]比较合理（对应频率0-10Hz）
        ω = np.random.uniform(0.1, 20 * np.pi, self.max_harmonics)
        
        # φ_i ∈ [0, T]
        φ = np.random.uniform(0, T, (self.n_servos, self.max_harmonics))
        
        return {
            'n': n,
            'T': T,
            'A': A,
            'ω': ω,
            'φ': φ
        }
    
    def get_fourier_curve(self, params, t):
        """计算在时间t的舵机角度"""
        n = params['n']
        T = params['T']
        A = params['A']
        ω = params['ω']
        φ = params['φ']
        
        # 确保t在合理范围内
        t = t % T  # 循环动作
        
        angles = np.zeros(self.n_servos)
        
        for servo_idx in range(self.n_servos):
            # 【修复】傅里叶级数公式：f(t) = A_0/2 + Σ_{k=1}^{n} A_k * cos(ω_k*t - φ_k)
            # 注意：应该从k=0到k=n-1，共n个谐波
            angle = A[servo_idx, 0] / 2.0
            
            # 【关键修复】range应该是(0, n)而不是(1, n)，这样n=1时也会有一个谐波
            for harmonic in range(0, n):
                angle += A[servo_idx, harmonic] * np.cos(
                    ω[harmonic] * t - φ[servo_idx, harmonic]
                )
            
            # 限制角度在合理范围内（假设舵机角度范围[-1, 1]对应实际角度范围）
            angles[servo_idx] = np.clip(angle, -1.0, 1.0)
        
        return angles
    
    def batch_get_fourier_curve(self, params, t_values):
        """批量计算多个时间点的舵机角度"""
        n = params['n']
        T = params['T']
        A = params['A']
        ω = params['ω']
        φ = params['φ']
        
        # t_values: [batch_size] 或标量
        t_values = np.array(t_values) % T
        
        if np.isscalar(t_values):
            t_values = np.array([t_values])
        
        batch_size = len(t_values)
        angles = np.zeros((batch_size, self.n_servos))
        
        for i, t in enumerate(t_values):
            for servo_idx in range(self.n_servos):
                angle = A[servo_idx, 0] / 2.0
                
                # 【修复】与get_fourier_curve保持一致，使用range(0, n)
                for harmonic in range(0, n):
                    angle += A[servo_idx, harmonic] * np.cos(
                        ω[harmonic] * t - φ[servo_idx, harmonic]
                    )
                
                angles[i, servo_idx] = np.clip(angle, -1.0, 1.0)
        
        return angles


class LMFModule(nn.Module):
    """
    低秩多模态融合模块：
    - 用于融合图像特征 x 和状态特征 state
    - 与 lmfGrasp 中的 LMFModule 保持一致，便于论文/代码对应
    """
    def __init__(self, input_dim1, input_dim2, hidden_dim, rank):
        super().__init__()
        self.rank = rank
        self.hidden_dim = hidden_dim

        self.fc_x_list = nn.ModuleList([
            nn.Linear(input_dim1, hidden_dim) for _ in range(rank)
        ])
        self.fc_s_list = nn.ModuleList([
            nn.Linear(input_dim2, hidden_dim) for _ in range(rank)
        ])

        self.fc_fusion = nn.Linear(rank * hidden_dim, hidden_dim)

    def forward(self, x, state):
        """
        x:     [B, input_dim1]
        state: [B, input_dim2]
        返回:  [B, hidden_dim]
        """
        batch_size = x.size(0)
        fusion_tensor = torch.zeros(batch_size, self.rank, self.hidden_dim, device=x.device)

        for i in range(self.rank):
            x_proj = self.fc_x_list[i](x)
            s_proj = self.fc_s_list[i](state)
            # Hadamard product
            fusion_tensor[:, i, :] = x_proj * s_proj

        fusion_flat = fusion_tensor.view(batch_size, -1)
        fused = self.fc_fusion(fusion_flat)
        return fused


class SpatioTemporalAttention(nn.Module):
    """
    来自 tsattenGrasp 的时空注意力融合模块：
    - 用注意力权重融合 CNN 特征与状态特征
    - 这里保留实现，方便在 PPO 中随时启用
    """
    def __init__(self, x_dim, state_dim, hidden_dim=200):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(x_dim, hidden_dim)
        self.key = nn.Linear(state_dim, hidden_dim)
        self.value = nn.Linear(state_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim + x_dim, hidden_dim)

    def forward(self, x, state):
        # x/state 均为 [feature_dim]，扩展 batch 维度以复用 tsattenGrasp 逻辑
        q = self.query(x.unsqueeze(0))
        k = self.key(state.unsqueeze(0))
        v = self.value(state.unsqueeze(0))

        scale = torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32, device=x.device))
        scores = torch.matmul(q, k.transpose(0, 1)) / scale
        attention_weights = F.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, v)

        combined = torch.cat([x, attended_values.squeeze(0)], dim=-1)
        return self.proj(combined)


class ActorCritic(nn.Module):
    def __init__(self, act_dim, node_num, use_fourier=False, n_servos=20, max_harmonics=20, T_max=10.0):
        super().__init__()
        self.node_num = node_num
        self.use_fourier = use_fourier
        
        # 保留原有的特征提取网络结构
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        
        self.fc0 = nn.Linear(in_features=6272, out_features=6000)
        self.fc1 = nn.Linear(in_features=6000, out_features=100)
        self.fc2 = nn.Linear(in_features=20, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=100)
        
        # 【新增】初始化卷积层和全连接层的权重，防止梯度爆炸
        # 使用更小的初始化值
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        for m in [self.fc0, self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(m.weight, gain=0.5)  # 降低gain从1.0到0.5
            nn.init.constant_(m.bias, 0)

        # 图像特征 + 状态特征 的 LMF 多模态融合模块
        # 与 lmfGrasp 中保持相同配置：100 + 100 -> 200
        self.lmf = LMFModule(input_dim1=100, input_dim2=100, hidden_dim=200, rank=5)

        # tsattenGrasp 时空注意力融合模块（调用在 forward 中默认注释）
        self.attention_fusion = SpatioTemporalAttention(x_dim=100, state_dim=100, hidden_dim=200)
        
        # 图神经网络部分
        self.conv_graph1 = torch_geometric.nn.GraphSAGE(1, 1000, 2, aggr='add')
        self.conv_graph2 = torch_geometric.nn.GATConv(1000, 1000, aggr='add')
        self.conv_graph3 = torch_geometric.nn.GraphSAGE(1000, 1000, 2, aggr='add')
        self.conv_graph4 = torch_geometric.nn.GATConv(1000, 1000, aggr='add')
        self.conv_graph5 = torch_geometric.nn.GCNConv(1000, 1000, 2, aggr='add')
        self.fc_graph = nn.Linear(1000, 100)
        
        # 共享特征层
        self.fc4 = nn.Linear(in_features=300, out_features=200)
        
        # === 【傅里叶模式】Actor头：输出傅里叶参数的分布 ===
        if self.use_fourier:
            self.n_servos = n_servos
            self.max_harmonics = max_harmonics
            self.T_max = T_max
            
            # 1. n_harmonics的logits（分类）
            self.n_head = nn.Linear(200, self.max_harmonics)
            
            # 2. T的参数（均值和log_std）
            self.T_head = nn.Linear(200, 2)  # [mean, log_std]
            
            # 3. A系列参数
            self.A_head = nn.Linear(200, self.n_servos * self.max_harmonics * 2)
            
            # 4. ω系列参数
            self.ω_head = nn.Linear(200, self.max_harmonics * 2)
            
            # 5. φ系列参数
            self.φ_head = nn.Linear(200, self.n_servos * self.max_harmonics * 2)
            
            # 初始化傅里叶头的权重（使用更小的初始化值，防止梯度爆炸）
            nn.init.orthogonal_(self.n_head.weight, gain=0.001)  # 进一步降低
            nn.init.constant_(self.n_head.bias, 0.0)
            nn.init.orthogonal_(self.T_head.weight, gain=0.001)
            nn.init.constant_(self.T_head.bias, 0.0)
            nn.init.orthogonal_(self.A_head.weight, gain=0.001)
            nn.init.constant_(self.A_head.bias, 0.0)
            nn.init.orthogonal_(self.ω_head.weight, gain=0.001)
            nn.init.constant_(self.ω_head.bias, 0.0)
            nn.init.orthogonal_(self.φ_head.weight, gain=0.001)
            nn.init.constant_(self.φ_head.bias, 0.0)
            
            # 【新增】初始化共享层的权重，防止梯度爆炸
            nn.init.orthogonal_(self.fc4.weight, gain=0.01)  # 降低gain
            nn.init.constant_(self.fc4.bias, 0.0)
            
        else:
            # === 【原始模式】Actor头：输出简单动作分布 ===
            self.actor_mu = nn.Sequential(
                nn.Linear(200, act_dim),
                nn.Tanh()  # Tanh激活函数将mu的范围限制在[-1, 1]
            )
            
            # 将log_sigma作为可学习的参数
            self.actor_log_sigma = nn.Parameter(torch.tensor([-0.5]))  # 初始sigma ≈ 0.61，增加探索
        
        # Critic头：输出状态值（两种模式共用）
        self.critic = nn.Linear(200, 1)
        
        # 【新增】初始化Critic头（在创建之后）
        nn.init.orthogonal_(self.critic.weight, gain=0.01)
        nn.init.constant_(self.critic.bias, 0.0)
    
    # 保留原有的图处理函数
    def create_edge_index(self):
        ans = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ]
        return torch.tensor(ans, dtype=torch.long)
    
    def creat_x(self, x_graph):
        ans = [[] for i in range(self.node_num)]
        for i in range(len(ans)):
            ans[i] = [x_graph[i]]
        return ans
    
    def creat_graph(self, x_graph):
        x = torch.as_tensor(self.creat_x(x_graph), dtype=torch.float32)
        edge_index = torch.as_tensor(self.create_edge_index(), dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)
        graph.x = graph.x.to(device)
        graph.edge_index = graph.edge_index.to(device)
        return graph

    def forward(self, x, state, x_graph):
        # 特征提取部分与原DQN相同
        self.graph = self.creat_graph(x_graph)
        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        
        # 调试：检查输入
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[ERROR] forward输入x包含NaN/Inf，使用零向量替代")
            x = torch.zeros_like(x)
        
        x = torch.unsqueeze(x, dim=0)
        x = self.conv1(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[ERROR] conv1后出现NaN/Inf")
            x = torch.zeros_like(x)
        x = self.relu(x)
        x = self.conv2(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[ERROR] conv2后出现NaN/Inf")
            x = torch.zeros_like(x)
        x = self.relu(x)
        x = self.conv3(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[ERROR] conv3后出现NaN/Inf")
            x = torch.zeros_like(x)
        x = x.view(x.size(0), -1)
        x = torch.flatten(x)
        x = self.fc0(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[ERROR] fc0后出现NaN/Inf")
            x = torch.zeros_like(x)
        x = self.fc1(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[ERROR] fc1后出现NaN/Inf")
            x = torch.zeros_like(x)
        
        # 【修复】更安全的归一化方法，防止NaN
        # 1. 先检查输入是否包含NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[ERROR] x在归一化前就包含NaN/Inf，使用零向量替代")
            normalized_data1 = torch.zeros_like(x)
        else:
            min_val1 = torch.min(x)
            max_val1 = torch.max(x)
            range1 = max_val1 - min_val1
            
            # 2. 检查range是否有效
            if torch.isnan(range1) or torch.isinf(range1) or range1 < 1e-6:
                print(f"[WARNING] x的range无效或太小({range1:.6f})，使用标准化替代归一化")
                # 使用标准化（减均值除标准差）替代归一化
                mean_val = torch.mean(x)
                std_val = torch.std(x)
                if std_val < 1e-6:
                    normalized_data1 = torch.zeros_like(x)
                else:
                    normalized_data1 = (x - mean_val) / (std_val + 1e-8)
            else:
                # 3. 正常归一化到[0, 1]
                normalized_data1 = (x - min_val1) / (range1 + 1e-8)
            
            # 4. 最后检查结果
            if torch.isnan(normalized_data1).any() or torch.isinf(normalized_data1).any():
                print(f"[ERROR] x归一化后出现NaN/Inf，使用零向量替代")
                normalized_data1 = torch.zeros_like(x)
        
        state = torch.as_tensor(state, dtype=torch.float32).to(device)
        state = self.fc2(state)
        state = self.fc3(state)
        
        # 【修复】state的安全归一化
        if torch.isnan(state).any() or torch.isinf(state).any():
            print(f"[ERROR] state在归一化前就包含NaN/Inf")
            normalized_data2 = torch.zeros_like(state)
        else:
            min_val2 = torch.min(state)
            max_val2 = torch.max(state)
            range2 = max_val2 - min_val2
            
            if torch.isnan(range2) or torch.isinf(range2) or range2 < 1e-6:
                # 使用标准化替代
                mean_val = torch.mean(state)
                std_val = torch.std(state)
                if std_val < 1e-6:
                    normalized_data2 = torch.zeros_like(state)
                else:
                    normalized_data2 = (state - mean_val) / (std_val + 1e-8)
            else:
                normalized_data2 = (state - min_val2) / (range2 + 1e-8)
            
            if torch.isnan(normalized_data2).any() or torch.isinf(normalized_data2).any():
                print(f"[ERROR] state归一化后出现NaN/Inf")
                normalized_data2 = torch.zeros_like(state)
        
        x_graph = self.creat_graph(x_graph)
        edge_index = x_graph.edge_index
        x_graph = self.conv_graph1(x_graph.x, edge_index)
        x_graph = self.relu(x_graph)
        x_graph = self.conv_graph2(x_graph, edge_index)
        x_graph = self.relu(x_graph)
        x_graph = self.conv_graph3(x_graph, edge_index)
        x_graph = self.relu(x_graph)
        x_graph = self.conv_graph4(x_graph, edge_index)
        x_graph = self.relu(x_graph)
        x_graph = self.conv_graph5(x_graph, edge_index)
        x_graph = torch.mean(x_graph, dim=0)
        x_graph = self.fc_graph(x_graph)

        # 【修复】x_graph的安全归一化
        if torch.isnan(x_graph).any() or torch.isinf(x_graph).any():
            print(f"[ERROR] x_graph在归一化前就包含NaN/Inf")
            normalized_x_graph = torch.zeros_like(x_graph)
        else:
            min_val3 = torch.min(x_graph)
            max_val3 = torch.max(x_graph)
            range3 = max_val3 - min_val3
            
            if torch.isnan(range3) or torch.isinf(range3) or range3 < 1e-6:
                # 使用标准化替代
                mean_val = torch.mean(x_graph)
                std_val = torch.std(x_graph)
                if std_val < 1e-6:
                    normalized_x_graph = torch.zeros_like(x_graph)
                else:
                    normalized_x_graph = (x_graph - mean_val) / (std_val + 1e-8)
            else:
                normalized_x_graph = (x_graph - min_val3) / (range3 + 1e-8)
            
            if torch.isnan(normalized_x_graph).any() or torch.isinf(normalized_x_graph).any():
                print(f"[ERROR] x_graph归一化后出现NaN/Inf")
                normalized_x_graph = torch.zeros_like(x_graph)

        # 使用 LMF 融合图像特征和状态特征（默认启用）
        img_feat = normalized_data1.unsqueeze(0)   # [1, 100]
        state_feat = normalized_data2.unsqueeze(0) # [1, 100]
        fused_feat = self.lmf(img_feat, state_feat).squeeze(0)  # [200]

        # 再与图特征拼接，得到最终 300 维特征
        state_x = torch.cat((fused_feat, normalized_x_graph), dim=-1)
        features = self.fc4(state_x)
        
        # 【修复】检查features是否包含NaN/Inf
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"[ERROR] features包含NaN/Inf，使用零向量替代")
            features = torch.zeros_like(features)
        
        # === 【傅里叶模式】输出傅里叶参数分布 ===
        if self.use_fourier:
            # n_harmonics分布（离散）
            n_logits = self.n_head(features)
            # 防止无效值
            n_logits = torch.clamp(n_logits, min=-10.0, max=10.0)
            
            # T分布
            T_params = self.T_head(features)
            T_mean = torch.sigmoid(T_params[0:1]) * self.T_max
            T_log_std = T_params[1:2].clamp(min=-2.0, max=0.0)
            T_std = torch.exp(T_log_std).clamp(min=0.01, max=1.0)
            
            # A分布
            A_params = self.A_head(features)
            A_params = A_params.view(self.n_servos, self.max_harmonics, 2)
            A_mean = torch.tanh(A_params[:, :, 0])  # [-1, 1]
            A_log_std = A_params[:, :, 1].clamp(min=-2.0, max=0.0)
            A_std = torch.exp(A_log_std).clamp(min=0.01, max=0.5)
            
            # ω分布
            ω_params = self.ω_head(features)
            ω_params = ω_params.view(self.max_harmonics, 2)
            ω_mean = F.softplus(ω_params[:, 0]) + 0.1  # (0, ∞)
            ω_mean = ω_mean.clamp(min=0.1, max=20*np.pi)
            ω_log_std = ω_params[:, 1].clamp(min=-2.0, max=0.0)
            ω_std = torch.exp(ω_log_std).clamp(min=0.01, max=1.0)
            
            # φ分布
            φ_params = self.φ_head(features)
            φ_params = φ_params.view(self.n_servos, self.max_harmonics, 2)
            φ_mean = φ_params[:, :, 0]  # 无约束，后续会取模
            φ_log_std = φ_params[:, :, 1].clamp(min=-2.0, max=0.0)
            φ_std = torch.exp(φ_log_std).clamp(min=0.01, max=1.0)
            
            # Critic: 输出状态值
            value = self.critic(features)
            
            return {
                'n_logits': n_logits,
                'T_mean': T_mean, 'T_std': T_std,
                'A_mean': A_mean, 'A_std': A_std,
                'ω_mean': ω_mean, 'ω_std': ω_std,
                'φ_mean': φ_mean, 'φ_std': φ_std
            }, value
        
        # === 【原始模式】输出简单动作分布 ===
        else:
            # Actor: 输出均值 mu
            mu = self.actor_mu(features)
            
            # 计算标准差 sigma
            log_sigma = self.actor_log_sigma.expand_as(mu)
            sigma = torch.exp(log_sigma)
            
            # 构建正态分布
            dist = Normal(mu, sigma)
            
            # Critic: 输出状态值
            value = self.critic(features)
            
            return dist, value

class PPO:
    def __init__(self, node_num, env_information, act_dim=1, use_fourier=True, 
                 n_servos=20, max_harmonics=20, T_max=10.0, step_interval=0.1):
        self.node_num = node_num
        self.env_information = env_information
        self.act_dim = act_dim
        self.use_fourier = use_fourier
        self.step_interval = step_interval
        self.T_max = T_max  # 保存T_max属性
        self.n_servos = n_servos  # 保存n_servos属性
        self.max_harmonics = max_harmonics  # 保存max_harmonics属性
        
        # PPO超参数
        self.gamma = 0.99  # 折扣因子
        self.gae_lambda = 0.95  # GAE参数
        self.clip_ratio = 0.2  # PPO裁剪参数（提高到0.2，允许更大的策略更新）
        self.value_coef = 0.5  # 值函数损失系数（提高，让值函数学习更快）
        self.entropy_coef = 0.01  # 熵系数
        self.policy_loss_scale = 1.0  # policy_loss缩放因子（提高到1.0）
        self.max_grad_norm = 0.5  # 梯度裁剪阈值（降低到0.5，更严格）

        # 学习率和优化器参数
        self.lr = 3e-5  # 【修复】进一步降低学习率到3e-5
        self.lr_decay = 0.9998  # 【修复】调整学习率衰减，更缓慢

        # PPO更新参数
        self.update_epochs = 4  # 更新次数（降低到4，避免过拟合）
        self.batch_size = 32  # 批大小（降低到32，更稳定）

        # 初始化策略网络
        self.policy = ActorCritic(
            act_dim=self.act_dim, 
            node_num=self.node_num,
            use_fourier=use_fourier,
            n_servos=n_servos,
            max_harmonics=max_harmonics,
            T_max=T_max
        ).to(device)
        
        # 初始化傅里叶动作空间（如果使用）
        if self.use_fourier:
            self.action_space = FourierActionSpace(
                n_servos=n_servos,
                max_harmonics=max_harmonics,
                T_max=T_max
            )

        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        
        # 当前 episode 的轨迹数据（每轮 episode 结束后会被打包进 episode_buffer）
        self.states = []
        self.actions = []          # 存储动作参数（傅里叶模式）或 tanh 后的动作（原始模式）
        self.actions_raw = []      # 存储原始动作（仅原始模式使用）
        self.rewards = []
        self.next_states = []
        self.values = []           # 合并存储价值
        self.log_probs = []       # 合并存储对数概率
        self.dones = []
        
        # 【傅里叶模式专用】存储额外信息
        if self.use_fourier:
            self.images = []       # 存储图像
            self.angles = []       # 存储舵机角度
            self.time_steps = []   # 存储时间步
        
        # 【新增】自适应探索率调整：记录学习历史（仅原始模式使用）
        if not self.use_fourier:
            self.learning_history = {
                'losses': deque(maxlen=20),           # 最近20个episode的总loss
                'policy_losses': deque(maxlen=20),    # 最近20个episode的policy_loss
                'rewards': deque(maxlen=20),          # 最近20个episode的累计reward
                'reward_sums': deque(maxlen=20),      # 最近20个episode的reward总和
            }
            # 探索率调整参数
            self.sigma_adjustment_rate = 0.02  # 每次调整的幅度（2%，更保守）
            self.sigma_min = 0.2   # 最小sigma
            self.sigma_max = 0.8   # 最大sigma
            self.min_episodes_before_adjust = 20  # 至少20个episode后才开始调整
            self.adjust_interval = 3  # 每3个episode调整一次，避免过于频繁
            self.last_adjust_episode = 0  # 记录上次调整的episode
    
    def choose_action(self, episode_num, obs, x_graph, explore=None):
        """
        选择动作
        - 傅里叶模式：生成整个周期的动作参数
        - 原始模式：生成单步动作
        """
        if isinstance(obs, tuple):
            x = obs[0]
            state = obs[1]
        else:
            x = obs
            state = x_graph

        # 确保所有输入都移到正确的设备
        if isinstance(x, torch.Tensor):
            x = x.to(device)
        else:
            x = torch.as_tensor(x, dtype=torch.float32).to(device)
        
        if isinstance(state, torch.Tensor):
            state = state.to(device)
        else:
            state = torch.as_tensor(state, dtype=torch.float32).to(device)

        # === 【傅里叶模式】生成傅里叶参数 ===
        if self.use_fourier:
            with torch.no_grad():
                dists, value = self.policy(x, state, x_graph)
                
                # 采样n_harmonics
                n_dist = Categorical(logits=dists['n_logits'])
                n_sample = n_dist.sample()
                n_sample = n_sample + 1  # 映射到[1, max_harmonics]
                
                # 采样T
                T_dist = Normal(dists['T_mean'], dists['T_std'])
                T_sample = T_dist.sample()
                T_sample = T_sample.clamp(0.1, self.action_space.T_max)
                
                # 采样A
                A_dist = Normal(dists['A_mean'], dists['A_std'])
                A_sample = A_dist.sample()
                A_sample = A_sample.clamp(-1.0, 1.0)
                
                # 采样ω
                ω_dist = Normal(dists['ω_mean'], dists['ω_std'])
                ω_sample = ω_dist.sample()
                ω_sample = ω_sample.clamp(0.1, 20 * np.pi)
                
                # 采样φ
                φ_dist = Normal(dists['φ_mean'], dists['φ_std'])
                φ_sample = φ_dist.sample()
                # 确保φ在[0, T]范围内
                φ_sample = φ_sample % T_sample
                
                # 计算log概率
                log_prob_n = n_dist.log_prob(n_sample - 1)
                log_prob_T = T_dist.log_prob(T_sample).sum()
                log_prob_A = A_dist.log_prob(A_sample).sum()
                log_prob_ω = ω_dist.log_prob(ω_sample).sum()
                log_prob_φ = φ_dist.log_prob(φ_sample).sum()
                
                # 总log概率
                total_log_prob = (log_prob_n + log_prob_T + 
                                log_prob_A + log_prob_ω + log_prob_φ)
                
                # 构建动作参数字典
                action_params = {
                    'n': n_sample.cpu().item(),
                    'T': T_sample.cpu().item(),
                    'A': A_sample.cpu().numpy(),
                    'ω': ω_sample.cpu().numpy(),
                    'φ': φ_sample.cpu().numpy()
                }
                
                return action_params, total_log_prob.item(), value.item(), None
        
        # === 【原始模式】生成单步动作 ===
        else:
            epsilon = max(0.05, 0.90 - episode_num * 0.001)
            if explore is not None:
                use_random = explore
            else:
                random_num = np.random.uniform()
                use_random = random_num < epsilon
                
            with torch.no_grad():
                dist, value = self.policy(x, state, x_graph)
                
                if use_random:
                    # 探索：根据智能体的动作维度生成随机动作
                    action_scaled = torch.tensor(np.random.uniform(-1, 1, size=self.act_dim), dtype=torch.float32).to(device)
                    action_raw = dist.sample()
                    action_scaled = torch.tanh(action_raw)
                else:
                    # 利用：从策略网络生成的分布中采样
                    action_raw = dist.sample()
                    action_scaled = torch.tanh(action_raw)
                
                # 正确计算 log_prob：需要减去 tanh 的雅可比修正项
                log_prob_raw = dist.log_prob(action_raw)
                tanh_correction = torch.log(1 - action_scaled.pow(2) + 1e-6)
                log_prob = (log_prob_raw - tanh_correction).sum(dim=-1)

                return action_scaled.cpu().numpy(), log_prob.item(), value.item(), action_raw.cpu().numpy()

    def store_transition_catch(self, state, action, reward, next_state, done, value, log_prob, action_raw=None, image=None, angles=None, time_step=None):
        """
        存储经验
        
        参数:
            state: 状态
            action: 动作参数（傅里叶模式）或 tanh 后的动作（原始模式）
            reward: 奖励
            next_state: 下一个状态
            done: 是否结束
            value: 价值估计
            log_prob: 对数概率
            action_raw: 原始动作（仅原始模式使用）
            image: 图像（仅傅里叶模式使用）
            angles: 舵机角度（仅傅里叶模式使用）
            time_step: 时间步（仅傅里叶模式使用）
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
        # 【傅里叶模式】存储额外信息
        if self.use_fourier:
            if image is not None:
                self.images.append(image)
            if angles is not None:
                self.angles.append(angles)
            if time_step is not None:
                self.time_steps.append(time_step)
        
        # 【原始模式】存储原始动作
        else:
            if action_raw is not None:
                self.actions_raw.append(action_raw)
            else:
                # 尝试从 tanh 后的动作反推原始动作（用于兼容性）
                action_tensor = torch.tensor(action, dtype=torch.float32).to(device)
                action_raw_approx = torch.atanh(torch.clamp(action_tensor, -0.9999, 0.9999))
                self.actions_raw.append(action_raw_approx.cpu().item())
    
    def calculate_advantages(self, rewards, values, dones):
        """
        计算优势函数和回报
        """
        if not rewards:          # 没有数据直接返回空
            return np.array([]), np.array([])

        # 将rewards和values转换为numpy数组以便处理
        if len(values) != len(rewards):
            print(f"警告: values 长度 ({len(values)}) 和 rewards 长度 ({len(rewards)}) 不匹配！这可能表明数据存储逻辑有误。")
            # 可以选择报错，或者截断到较短的那个长度（不推荐）
            # 这里选择报错，让开发者定位问题
            raise ValueError("Critical Error: self.values and self.rewards have different lengths.")

        values = np.array(values) 
        rewards = np.array(rewards)
        dones = np.array(dones)

        # 计算GAE优势函数
        advantages = np.zeros_like(rewards)
        last_advantage = 0

        # 从后向前计算优势函数
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # 对于最后一个时间步，使用0作为下一个值的估计
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        # 计算回报
        returns = advantages + values

        # 【新增】记录reward和advantages的统计信息，帮助诊断问题
        rewards_sum = rewards.sum()
        rewards_mean = rewards.mean()
        rewards_std = rewards.std()
        advantages_mean_before = advantages.mean()
        advantages_std_before = advantages.std()
        
        # 【关键修复】改进优势函数标准化策略
        # 问题分析：
        # 1. 完全标准化会丢失奖励的绝对大小信息
        # 2. 只中心化会保留尺度，但可能导致梯度过大
        # 3. 需要在两者之间找到平衡
        
        # 策略：使用自适应标准化
        advantages_mean = advantages.mean()
        advantages_std = advantages.std()
        
        # 如果标准差很小（奖励变化不大），只中心化
        if advantages_std < 1.0:
            advantages = advantages - advantages_mean
            print(f"  【优势函数】标准差小({advantages_std:.2f})，只中心化")
        # 如果标准差适中，使用温和的标准化
        elif advantages_std < 50.0:
            advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
            print(f"  【优势函数】标准差适中({advantages_std:.2f})，使用标准标准化")
        # 如果标准差很大，使用更温和的标准化（保留更多原始信息）
        else:
            # 使用平方根标准化，减少极端值的影响
            scale_factor = np.sqrt(advantages_std)
            advantages = (advantages - advantages_mean) / (scale_factor + 1e-8)
            print(f"  【优势函数】标准差大({advantages_std:.2f})，使用温和标准化(scale={scale_factor:.2f})")
        
        # 【新增】打印统计信息
        advantages_mean_after = advantages.mean()
        advantages_std_after = advantages.std()
        print(f"  【Reward统计】sum={rewards_sum:.2f}, mean={rewards_mean:.2f}, std={rewards_std:.2f}")
        print(f"  【Advantages】标准化前: mean={advantages_mean_before:.2f}, std={advantages_std_before:.2f}")
        print(f"  【Advantages】标准化后: mean={advantages_mean_after:.2f}, std={advantages_std_after:.2f}")

        return advantages, returns

    def get_current_sigma(self):
        """获取当前探索率（仅原始模式）"""
        if not self.use_fourier:
            return torch.exp(self.policy.actor_log_sigma).item()
        return None
    
    def _adjust_exploration_rate(self, episode_num):
        """
        根据学习情况自适应调整探索率（sigma）（仅原始模式）
        """
        # 仅在原始模式下调整
        if self.use_fourier:
            return
            
        # 检查是否满足调整条件
        if len(self.learning_history['losses']) < 10:  # 需要至少10个episode的数据
            return
        
        # 检查调整间隔
        if episode_num - self.last_adjust_episode < self.adjust_interval:
            return
        
        # 早期训练阶段保护：前N个episode保持高探索率
        if episode_num < self.min_episodes_before_adjust:
            return
        
        losses = list(self.learning_history['losses'])
        policy_losses = list(self.learning_history['policy_losses'])
        rewards = list(self.learning_history['reward_sums'])
        
        # 计算最近10个episode的趋势（使用更多数据更稳定）
        recent_losses = losses[-10:]
        recent_policy_losses = policy_losses[-10:]
        recent_rewards = rewards[-10:]
        
        # 1. 分析loss趋势
        loss_trend = (recent_losses[-1] - recent_losses[0]) / max(abs(recent_losses[0]), 1e-6)
        # 2. 分析reward趋势和绝对值
        reward_mean = sum(recent_rewards) / len(recent_rewards)
        reward_std = (sum((x - reward_mean)**2 for x in recent_rewards) / len(recent_rewards))**0.5
        reward_trend = (recent_rewards[-1] - recent_rewards[0]) / max(abs(recent_rewards[0]), 1e-6) if recent_rewards[0] != 0 else 0
        # 3. 分析policy_loss的平均值和稳定性
        policy_loss_mean = sum(recent_policy_losses) / len(recent_policy_losses)
        policy_loss_std = (sum((x - policy_loss_mean)**2 for x in recent_policy_losses) / len(recent_policy_losses))**0.5
        
        # 获取当前的sigma
        current_log_sigma = self.policy.actor_log_sigma.item()
        current_sigma = torch.exp(self.policy.actor_log_sigma).item()
        
        # 决策：是否应该调整探索率
        should_decrease = False  # 是否应该降低探索率
        should_increase = False  # 是否应该增加探索率
        reason = ""
        
        # 【优先】情况1：如果return为负或波动很大，增加探索率
        if reward_mean < 0 or reward_std > 100:
            should_increase = True
            reason = f"return为负({reward_mean:.2f})或波动大(std={reward_std:.2f})，需要更多探索"
        
        # 情况2：如果return持续为正且稳定上升，可以考虑降低探索率
        elif reward_mean > 50 and reward_trend > 0.1 and reward_std < 50:
            # 进一步检查：loss是否也在下降
            if loss_trend < -0.05:
                should_decrease = True
                reason = f"return为正且稳定上升(mean={reward_mean:.2f}, trend={reward_trend:.2%})，loss下降"
        
        # 情况3：loss持续下降且reward上升（更严格的条件）
        elif loss_trend < -0.15 and reward_trend > 0.15 and reward_mean > 0:
            should_decrease = True
            reason = f"loss大幅下降({loss_trend:.2%})且reward大幅上升({reward_trend:.2%})"
        
        # 情况4：loss上升或reward下降，增加探索率
        elif loss_trend > 0.15 or (reward_trend < -0.15 and reward_mean < 0):
            should_increase = True
            reason = f"loss上升({loss_trend:.2%})或reward下降({reward_trend:.2%})"
        
        # 情况5：policy_loss很小且稳定，且return为正（更严格的条件）
        elif policy_loss_mean < 0.3 and policy_loss_std < 0.15 and reward_mean > 30:
            should_decrease = True
            reason = f"policy_loss很小且稳定(mean={policy_loss_mean:.3f}, std={policy_loss_std:.3f})，return为正"
        
        # 情况6：policy_loss很大且不稳定，需要更多探索
        elif policy_loss_mean > 2.5 and policy_loss_std > 1.2:
            should_increase = True
            reason = f"policy_loss大且不稳定(mean={policy_loss_mean:.3f}, std={policy_loss_std:.3f})"
        
        # 执行调整
        if should_decrease and current_sigma > self.sigma_min:
            # 降低探索率：减小sigma（减小log_sigma）
            new_log_sigma = current_log_sigma - self.sigma_adjustment_rate
            new_sigma = torch.exp(torch.tensor(new_log_sigma)).item()
            if new_sigma >= self.sigma_min:
                self.policy.actor_log_sigma.data = torch.tensor([new_log_sigma], device=self.policy.actor_log_sigma.device)
                self.last_adjust_episode = episode_num
                print(f"  【自适应探索率】降低探索率: {reason}, sigma: {current_sigma:.4f} -> {new_sigma:.4f}")
        
        elif should_increase and current_sigma < self.sigma_max:
            # 增加探索率：增大sigma（增大log_sigma）
            new_log_sigma = current_log_sigma + self.sigma_adjustment_rate
            new_sigma = torch.exp(torch.tensor(new_log_sigma)).item()
            if new_sigma <= self.sigma_max:
                self.policy.actor_log_sigma.data = torch.tensor([new_log_sigma], device=self.policy.actor_log_sigma.device)
                self.last_adjust_episode = episode_num
                print(f"  【自适应探索率】增加探索率: {reason}, sigma: {current_sigma:.4f} -> {new_sigma:.4f}")
    
    def learn(self):
        """
        学习函数：每次调用时学习当前累积的所有episode数据
        - 调用端已控制每10个episode才调用一次
        - 直接使用 self.states 等列表中已累积的数据进行学习
        """
        # 检查是否有数据
        if len(self.rewards) == 0:
            print("  警告：没有数据可学习，返回0")
            return 0.0
        
        print(f"  【开始学习】使用累积的 {len(self.rewards)} 个样本进行学习")
        
        # 直接使用已累积的数据（已包含10个episode的数据）
        all_states = self.states
        all_actions = self.actions
        all_rewards = self.rewards
        all_next_states = self.next_states
        all_values = self.values
        all_log_probs = self.log_probs
        all_dones = self.dones

        # 计算优势函数和回报
        advantages, returns = self.calculate_advantages(all_rewards, all_values, all_dones)
        if len(advantages) == 0:
            return 0.0

        # === 【傅里叶模式】学习逻辑 ===
        if self.use_fourier:
            # 转换为张量
            batch_advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
            batch_returns = torch.tensor(returns, dtype=torch.float32).to(device)
            batch_log_probs = torch.tensor(all_log_probs, dtype=torch.float32).to(device)
            
            total_loss = 0
            total_policy_loss = 0
            batch_count = 0
            
            for _ in range(self.update_epochs):
                # 生成随机索引
                indices = torch.randperm(len(all_states))

                # 分批处理数据
                for start_idx in range(0, len(all_states), self.batch_size):
                    batch_indices = indices[start_idx:start_idx + self.batch_size]
                    
                    batch_x, batch_state, batch_x_graph = [], [], []
                    for idx in batch_indices:
                        if idx < len(all_states):
                            batch_x.append(all_states[idx][0])
                            batch_state.append(all_states[idx][1])
                            batch_x_graph.append(all_states[idx][2])
                            
                    if not batch_x:
                        continue

                    # 前向传播
                    dist_batch_values = [self.policy(x, s, g) for x, s, g in zip(batch_x, batch_state, batch_x_graph)]
                    
                    # 提取分布和值
                    dists_list = [dv[0] for dv in dist_batch_values]
                    values = torch.cat([dv[1].unsqueeze(0) for dv in dist_batch_values])
                    
                    # 检查分布是否包含无效值
                    for idx, dists in enumerate(dists_list):
                        if torch.isnan(dists['n_logits']).any():
                            print(f"警告: batch {start_idx}, sample {idx} 的 n_logits 包含 NaN")
                        if torch.isinf(dists['n_logits']).any():
                            print(f"警告: batch {start_idx}, sample {idx} 的 n_logits 包含 Inf")
                        if torch.isnan(dists['T_mean']).any() or torch.isnan(dists['T_std']).any():
                            print(f"警告: batch {start_idx}, sample {idx} 的 T 参数包含 NaN")
                        if torch.isnan(dists['A_mean']).any() or torch.isnan(dists['A_std']).any():
                            print(f"警告: batch {start_idx}, sample {idx} 的 A 参数包含 NaN")

                    # 获取当前批次的数据
                    batch_indices_int = batch_indices.cpu().numpy() if isinstance(batch_indices, torch.Tensor) else batch_indices
                    batch_log_probs_curr = batch_log_probs[batch_indices]
                    batch_advantages_curr = batch_advantages[batch_indices]
                    batch_returns_curr = batch_returns[batch_indices]
                    
                    # 重新计算新策略的概率
                    policy_loss = 0
                    entropy = 0
                    
                    for i in range(len(batch_x)):
                        dists = dists_list[i]
                        
                        # 从存储的动作参数中获取
                        action_params = all_actions[int(batch_indices_int[i])]
                        
                        # 检查并修复无效值
                        n_logits_clean = dists['n_logits'].clone()
                        if torch.isnan(n_logits_clean).any() or torch.isinf(n_logits_clean).any():
                            print(f"警告: n_logits包含无效值，使用均匀分布替代")
                            n_logits_clean = torch.zeros_like(n_logits_clean)
                        
                        # 重新计算log概率
                        n_dist = Categorical(logits=n_logits_clean)
                        T_dist = Normal(dists['T_mean'].clamp(0.1, self.T_max), dists['T_std'].clamp(min=0.01))
                        A_dist = Normal(dists['A_mean'].clamp(-1.0, 1.0), dists['A_std'].clamp(min=0.01))
                        ω_dist = Normal(dists['ω_mean'].clamp(0.1, 20*np.pi), dists['ω_std'].clamp(min=0.01))
                        φ_dist = Normal(dists['φ_mean'], dists['φ_std'].clamp(min=0.01))
                        
                        # 转换为tensor
                        n_tensor = torch.tensor(action_params['n'] - 1, dtype=torch.long).to(device)
                        T_tensor = torch.tensor(action_params['T'], dtype=torch.float32).to(device)
                        A_tensor = torch.tensor(action_params['A'], dtype=torch.float32).to(device)
                        ω_tensor = torch.tensor(action_params['ω'], dtype=torch.float32).to(device)
                        φ_tensor = torch.tensor(action_params['φ'], dtype=torch.float32).to(device)
                        
                        # 计算新的log概率
                        new_log_prob = (
                            n_dist.log_prob(n_tensor) +
                            T_dist.log_prob(T_tensor).sum() +
                            A_dist.log_prob(A_tensor).sum() +
                            ω_dist.log_prob(ω_tensor).sum() +
                            φ_dist.log_prob(φ_tensor).sum()
                        )
                        
                        # 计算ratio
                        ratio = torch.exp(new_log_prob - batch_log_probs_curr[i])
                        
                        # PPO裁剪
                        surr1 = ratio * batch_advantages_curr[i]
                        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages_curr[i]
                        
                        policy_loss += -torch.min(surr1, surr2)
                        
                        # 计算熵
                        entropy += n_dist.entropy()

                    policy_loss = policy_loss / len(batch_x)
                    entropy = entropy / len(batch_x)
                    
                    total_policy_loss += policy_loss.item()
                    batch_count += 1

                    # 缩放policy_loss
                    policy_loss_scaled = policy_loss * self.policy_loss_scale

                    # 值函数损失
                    value_loss = nn.MSELoss()(values, batch_returns_curr)
                    value_loss = torch.clamp(value_loss, max=100.0)

                    # 总损失
                    loss = policy_loss_scaled + self.value_coef * value_loss - self.entropy_coef * entropy

                    # 优化步骤
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # 【修复】检查梯度是否包含NaN/Inf
                    has_nan_grad = False
                    for name, param in self.policy.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"[ERROR] 参数 {name} 的梯度包含NaN/Inf")
                                has_nan_grad = True
                                # 将NaN/Inf梯度置零
                                param.grad[torch.isnan(param.grad)] = 0
                                param.grad[torch.isinf(param.grad)] = 0
                    
                    if has_nan_grad:
                        print(f"[WARNING] 检测到NaN/Inf梯度，已清零，跳过本次更新")
                        continue
                    
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                    total_loss += loss.item()

            # 更新学习率
            self.scheduler.step()
            
            # 清空轨迹数据
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.next_states.clear()
            self.dones.clear()
            self.values.clear()
            self.log_probs.clear()
            self.images.clear()
            self.angles.clear()
            self.time_steps.clear()
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  【学习完成】已清空数据，等待下一轮10个episode...")
            print("total_loss:", total_loss)
            return total_loss / self.update_epochs
        
        # === 【原始模式】学习逻辑 ===
        else:
            all_actions_raw = self.actions_raw
            
            # 将数据转换为张量
            batch_states = all_states
            batch_advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
            batch_returns = torch.tensor(returns, dtype=torch.float32).to(device)
            batch_actions = torch.tensor(all_actions, dtype=torch.float32).to(device)
            batch_log_probs = torch.tensor(all_log_probs, dtype=torch.float32).to(device)
            
            # 确保 actions_raw 列表存在且长度匹配
            actions_raw_list = list(all_actions_raw)
            if len(actions_raw_list) != len(all_actions):
                # 如果长度不匹配，从 actions 反推（向后兼容）
                actions_raw_list = [
                    torch.atanh(torch.clamp(torch.tensor(a), -0.9999, 0.9999)).item()
                    for a in all_actions
                ]
            
            total_loss = 0
            total_policy_loss = 0
            batch_count = 0
            
            for _ in range(self.update_epochs):
                # 生成随机索引
                indices = torch.randperm(len(batch_states))

                # 分批处理数据
                for start_idx in range(0, len(batch_states), self.batch_size):
                    batch_indices = indices[start_idx:start_idx + self.batch_size]
                    
                    batch_x, batch_state, batch_x_graph = [], [], []
                    for idx in batch_indices:
                        if idx < len(batch_states):
                            batch_x.append(batch_states[idx][0])
                            batch_state.append(batch_states[idx][1])
                            batch_x_graph.append(batch_states[idx][2])
                            
                    if not batch_x:
                        continue

                    # 前向传播
                    dist_batch_values = [self.policy(x, s, g) for x, s, g in zip(batch_x, batch_state, batch_x_graph)]
                    
                    # 提取分布和值
                    dists = [dv[0] for dv in dist_batch_values]
                    values = torch.cat([dv[1].unsqueeze(0) for dv in dist_batch_values])

                    # 获取当前批次的动作、对数概率、优势、回报等
                    batch_indices_int = batch_indices.cpu().numpy() if isinstance(batch_indices, torch.Tensor) else batch_indices
                    batch_actions_curr = batch_actions[batch_indices]
                    batch_log_probs_curr = batch_log_probs[batch_indices]
                    batch_advantages_curr = batch_advantages[batch_indices]
                    batch_returns_curr = batch_returns[batch_indices]
                    
                    # 计算新的对数概率和熵
                    batch_actions_raw = torch.tensor([actions_raw_list[int(idx)] for idx in batch_indices_int], 
                                                      dtype=torch.float32).to(device)
                    
                    policy_loss = 0
                    entropy = 0
                    
                    for i in range(len(batch_x)):
                        action_raw_i = batch_actions_raw[i]
                        if action_raw_i.dim() == 0:
                            action_raw_i = action_raw_i.unsqueeze(0)
                        
                        # 计算原始分布的对数概率
                        new_log_prob_raw = dists[i].log_prob(action_raw_i)
                        
                        # 计算 tanh 后的动作
                        action_tanh_i = torch.tanh(action_raw_i)
                        # 应用 tanh squashing 修正
                        tanh_correction = torch.log(1 - action_tanh_i.pow(2) + 1e-6)
                        new_log_prob = (new_log_prob_raw - tanh_correction).sum(dim=-1)
                        
                        ratio = torch.exp(new_log_prob - batch_log_probs_curr[i])
                        
                        surr1 = ratio * batch_advantages_curr[i]
                        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages_curr[i]
                        
                        policy_loss += -torch.min(surr1, surr2)
                        entropy += dists[i].entropy()

                    policy_loss = policy_loss / len(batch_x)
                    entropy = entropy / len(batch_x)
                    
                    total_policy_loss += policy_loss.item()
                    batch_count += 1

                    # 缩放policy_loss
                    policy_loss_scaled = policy_loss * self.policy_loss_scale

                    # 值函数损失
                    value_loss = nn.MSELoss()(values, batch_returns_curr)
                    value_loss = torch.clamp(value_loss, max=100.0)

                    # 总损失
                    loss = policy_loss_scaled + self.value_coef * value_loss - self.entropy_coef * entropy

                    if start_idx == 0:
                        print(f"  Loss分解: policy_loss(原始)={policy_loss.item():.4f}, policy_loss(缩放后)={policy_loss_scaled.item():.4f}")
                        print(f"  Loss分解: value_loss={value_loss.item():.4f}, entropy={entropy.item():.4f}")
                        print(f"  总loss={loss.item():.4f}")

                    # 优化步骤
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # 【修复】检查梯度是否包含NaN/Inf
                    has_nan_grad = False
                    for name, param in self.policy.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"[ERROR] 参数 {name} 的梯度包含NaN/Inf")
                                has_nan_grad = True
                                # 将NaN/Inf梯度置零
                                param.grad[torch.isnan(param.grad)] = 0
                                param.grad[torch.isinf(param.grad)] = 0
                    
                    if has_nan_grad:
                        print(f"[WARNING] 检测到NaN/Inf梯度，已清零，跳过本次更新")
                        continue
                    
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                    total_loss += loss.item()

            # 更新学习率
            self.scheduler.step()
            
            # 记录当前episode的学习指标
            avg_loss = total_loss / self.update_epochs
            reward_sum = sum(all_rewards) if all_rewards else 0.0
            policy_loss_avg = total_policy_loss / batch_count if batch_count > 0 else 0.0
            
            self.learning_history['losses'].append(avg_loss)
            self.learning_history['policy_losses'].append(policy_loss_avg)
            self.learning_history['rewards'].append(reward_sum)
            self.learning_history['reward_sums'].append(reward_sum)
            
            # 根据学习情况自适应调整探索率
            episode_num = len(self.learning_history['losses'])
            self._adjust_exploration_rate(episode_num)

            # 学习完成，清空轨迹数据
            self.states.clear()
            self.actions.clear()
            self.actions_raw.clear()
            self.rewards.clear()
            self.next_states.clear()
            self.dones.clear()
            self.values.clear()
            self.log_probs.clear()
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  【学习完成】已清空数据，等待下一轮10个episode...")
            print("total_loss:", total_loss)
            return total_loss / self.update_epochs