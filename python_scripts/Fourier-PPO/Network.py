import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal
from FourierActionSpace import FourierActionSpace


# ===================== Actor网络 =====================
class FourierActorNetwork(nn.Module):
    """
    Actor网络：输出傅里叶参数的分布
    """
    def __init__(self, state_dim, action_space: FourierActionSpace, hidden_dim=512):
        super().__init__()
        
        self.action_space = action_space
        self.max_harmonics = action_space.max_harmonics
        self.n_servos = action_space.n_servos
        
        # 基础网络
        self.base_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 输出头
        # 1. n_harmonics的logits（分类）
        self.n_head = nn.Linear(hidden_dim, self.max_harmonics)
        
        # 2. T的参数（均值和log_std）
        self.T_head = nn.Linear(hidden_dim, 2)  # [mean, log_std]
        
        # 3. A系列参数
        self.A_head = nn.Linear(hidden_dim, self.n_servos * self.max_harmonics * 2)
        
        # 4. ω系列参数
        self.ω_head = nn.Linear(hidden_dim, self.max_harmonics * 2)
        
        # 5. φ系列参数
        self.φ_head = nn.Linear(hidden_dim, self.n_servos * self.max_harmonics * 2)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        features = self.base_net(state)
        
        # n_harmonics分布（离散）
        n_logits = self.n_head(features)
        
        # T分布
        T_params = self.T_head(features)
        T_mean = torch.sigmoid(T_params[:, 0:1]) * self.action_space.T_max
        T_log_std = T_params[:, 1:2]
        T_std = torch.exp(T_log_std).clamp(min=0.01, max=1.0)
        
        # A分布
        A_params = self.A_head(features)
        A_params = A_params.view(-1, self.n_servos, self.max_harmonics, 2)
        A_mean = torch.tanh(A_params[:, :, :, 0])  # [-1, 1]
        A_log_std = A_params[:, :, :, 1]
        A_std = torch.exp(A_log_std).clamp(min=0.01, max=0.5)
        
        # ω分布
        ω_params = self.ω_head(features)
        ω_params = ω_params.view(-1, self.max_harmonics, 2)
        ω_mean = F.softplus(ω_params[:, :, 0]) + 0.1  # (0, ∞)
        ω_log_std = ω_params[:, :, 1]
        ω_std = torch.exp(ω_log_std).clamp(min=0.01, max=1.0)
        
        # φ分布
        φ_params = self.φ_head(features)
        φ_params = φ_params.view(-1, self.n_servos, self.max_harmonics, 2)
        φ_mean = φ_params[:, :, :, 0]  # 无约束，后续会取模
        φ_log_std = φ_params[:, :, :, 1]
        φ_std = torch.exp(φ_log_std).clamp(min=0.01, max=1.0)
        
        return {
            'n_logits': n_logits,
            'T_mean': T_mean, 'T_log_std': T_log_std, 'T_std': T_std,
            'A_mean': A_mean, 'A_log_std': A_log_std, 'A_std': A_std,
            'ω_mean': ω_mean, 'ω_log_std': ω_log_std, 'ω_std': ω_std,
            'φ_mean': φ_mean, 'φ_log_std': φ_log_std, 'φ_std': φ_std
        }
    
    def sample(self, state, deterministic=False):
        """
        从分布中采样参数
        deterministic=True: 使用均值（用于评估）
        deterministic=False: 采样（用于训练）
        """
        dists = self.forward(state)
        batch_size = state.shape[0]
        
        # 采样n_harmonics
        n_dist = Categorical(logits=dists['n_logits'])
        if deterministic:
            n_sample = torch.argmax(dists['n_logits'], dim=1)
        else:
            n_sample = n_dist.sample()
        n_sample = n_sample + 1  # 映射到[1, max_harmonics]
        
        # 采样T
        T_dist = Normal(dists['T_mean'], dists['T_std'])
        if deterministic:
            T_sample = dists['T_mean']
        else:
            T_sample = T_dist.sample()
        T_sample = T_sample.clamp(0.1, self.action_space.T_max)
        
        # 采样A
        A_dist = Normal(dists['A_mean'], dists['A_std'])
        if deterministic:
            A_sample = dists['A_mean']
        else:
            A_sample = A_dist.sample()
        A_sample = A_sample.clamp(-1.0, 1.0)
        
        # 采样ω
        ω_dist = Normal(dists['ω_mean'], dists['ω_std'])
        if deterministic:
            ω_sample = dists['ω_mean']
        else:
            ω_sample = ω_dist.sample()
        ω_sample = ω_sample.clamp(0.1, 20 * np.pi)
        
        # 采样φ
        φ_dist = Normal(dists['φ_mean'], dists['φ_std'])
        if deterministic:
            φ_sample = dists['φ_mean']
        else:
            φ_sample = φ_dist.sample()
        # 确保φ在[0, T]范围内
        φ_sample = φ_sample % T_sample.unsqueeze(-1).unsqueeze(-1)
        
        # 计算log概率
        log_probs = {}
        if not deterministic:
            log_probs['n'] = n_dist.log_prob(n_sample - 1)
            log_probs['T'] = T_dist.log_prob(T_sample).sum(dim=1)
            log_probs['A'] = A_dist.log_prob(A_sample).sum(dim=[1, 2])
            log_probs['ω'] = ω_dist.log_prob(ω_sample).sum(dim=1)
            log_probs['φ'] = φ_dist.log_prob(φ_sample).sum(dim=[1, 2])
            
            # 总log概率
            total_log_prob = (log_probs['n'] + log_probs['T'] + 
                            log_probs['A'] + log_probs['ω'] + log_probs['φ'])
        else:
            total_log_prob = None
        
        return {
            'n': n_sample,
            'T': T_sample,
            'A': A_sample,
            'ω': ω_sample,
            'φ': φ_sample
        }, total_log_prob, dists

# ===================== Critic网络 =====================
class CriticNetwork(nn.Module):
    """Critic网络：评估状态价值"""
    def __init__(self, state_dim, hidden_dim=512):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        return self.net(state)
