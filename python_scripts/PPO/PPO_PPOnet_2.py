import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from python_scripts.Project_config import path_list, BATCH_SIZE, LR, EPSILON, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, device, gps_goal, gps_goal1
import torch_geometric
from torch_geometric.data import Data
from torch.distributions import Normal

# PPO网络结构，包含策略网络和价值网络
class ActorCritic(nn.Module):
    def __init__(self, act_dim, node_num):
        """
        抬腿训练使用的ActorCritic网络
        Args:
            act_dim: 动作维度，抬腿训练中通常为8个动作
            node_num: 图节点数量
        """
        super().__init__()
        self.node_num = node_num
        
        # 图像特征提取网络
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        
        # 图像特征全连接层
        self.fc0 = nn.Linear(in_features=6272, out_features=6000)
        self.fc1 = nn.Linear(in_features=6000, out_features=100)
        
        # 状态特征处理层
        self.fc2 = nn.Linear(in_features=20, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=100)
        
        # 图神经网络部分，用于处理关节角度
        self.conv_graph1 = torch_geometric.nn.GraphSAGE(1, 1000, 2, aggr='add')
        self.conv_graph2 = torch_geometric.nn.GATConv(1000, 1000, heads=1, aggr='add')
        self.conv_graph3 = torch_geometric.nn.GraphSAGE(1000, 1000, 2, aggr='add')
        self.conv_graph4 = torch_geometric.nn.GATConv(1000, 1000, heads=1, aggr='add')
        self.conv_graph5 = torch_geometric.nn.GraphSAGE(1000, 1000, 2, aggr='add')
        self.fc_graph = nn.Linear(1000, 100)
        
        # 共享特征层
        self.fc4 = nn.Linear(in_features=300, out_features=200)
        
        # Actor头：输出动作概率
        self.actor_mu = nn.Sequential(
            nn.Linear(200, act_dim),
            nn.Tanh()  # Tanh激活函数将mu的范围限制在[-1, 1]
        )
        # 初始化log_sigma为更小的值
        self.actor_log_sigma = nn.Parameter(torch.zeros(act_dim) * 0.1)  # 可学习的对数标准差
        
        # Critic头：输出状态值
        self.critic = nn.Linear(200, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def create_edge_index(self):
        """创建图边索引"""
        ans = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ]
        return torch.tensor(ans, dtype=torch.long)
    
    def creat_x(self, x_graph):
        """从输入创建图节点特征"""
        ans = [[] for i in range(self.node_num)]
        for i in range(len(ans)):
            ans[i] = [x_graph[i]]
        return ans
    
    def creat_graph(self, x_graph):
        """创建图结构"""
        x = torch.as_tensor(self.creat_x(x_graph), dtype=torch.float32)
        edge_index = torch.as_tensor(self.create_edge_index(), dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)
        graph.x = graph.x.to(device)
        graph.edge_index = graph.edge_index.to(device)
        return graph

    def forward(self, x, state, x_graph):
        """
        前向传播
        Args:
            x: 图像数据
            state: 状态数据（关节角度等）
            x_graph: 图节点特征数据
        """
        # 创建图
        self.graph = self.creat_graph(x_graph)
        
        # 图像特征处理
        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        x = torch.unsqueeze(x, dim=0)
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = torch.flatten(x)
        x = self.fc0(x)
        x = self.fc1(x)
        
        # 特征归一化
        min_val1 = torch.min(x)
        max_val1 = torch.max(x)
        normalized_data1 = torch.div(torch.sub(x, min_val1), torch.sub(max_val1, min_val1))
        # 确保normalized_data1是2维的
        if len(normalized_data1.shape) == 1:
            normalized_data1 = normalized_data1.unsqueeze(0)
        
        # 处理状态数据
        if isinstance(state, (list, np.ndarray)):
            state = torch.FloatTensor(np.array(state)).to(device)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, state_dim)
        
        # # 处理 state 的数据维度，确保其为 1 * 4
        # if state.shape[1] > 4:
        #     state = state[:, :4]  # 取前四个数据
        # elif state.shape[1] < 4:
        #     # 计算需要补零的数量
        #     padding_size = 4 - state.shape[1]
        #     # 在最后一个维度上补零
        #     state = torch.nn.functional.pad(state, (0, padding_size), mode='constant', value=0)
        # # 确保最终维度为 1 * 4
    # assert state.shape == (1, 4), f"Unexpected state shape after processing: {state.shape}"

        state = self.fc2(state)
        state = self.fc3(state)
        min_val2 = torch.min(state)
        max_val2 = torch.max(state)
        normalized_data2 = torch.div(torch.sub(state, min_val2), torch.sub(max_val2, min_val2))
        
        # 图神经网络处理
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
        
        # 特征归一化
        min_val3 = torch.min(x_graph)
        max_val3 = torch.max(x_graph)
        normalized_x_graph = torch.div(torch.sub(x_graph, min_val3), torch.sub(max_val3, min_val3))
        # 确保normalized_x_graph是2维的
        if len(normalized_x_graph.shape) == 1:
            normalized_x_graph = normalized_x_graph.unsqueeze(0)
        
        # 特征融合
        state_x = torch.cat((normalized_data1, normalized_data2, normalized_x_graph), dim=-1)
        features = self.fc4(state_x)
        
        # 添加数值检查
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("警告:特征中出现NaN或Inf，使用默认分布--------185")
            mu = torch.zeros_like(self.actor_mu(torch.zeros_like(features))).to(device)
            sigma = torch.ones_like(mu).to(device) * 0.1
            dist = Normal(mu, sigma)
            value = torch.zeros(1).to(device)
            return dist, value
        
        # Actor: 输出动作分布的参数
        with torch.cuda.amp.autocast(enabled=True):
            mu = self.actor_mu(features)
            log_sigma = self.actor_log_sigma.expand_as(mu)
            sigma = torch.exp(log_sigma) + 1e-6
            
            # 数值检查
            if torch.isnan(mu).any() or torch.isinf(mu).any() or \
               torch.isnan(sigma).any() or torch.isinf(sigma).any():
                print("警告:动作分布中出现NaN或Inf，使用默认分布-----------201")
                mu = torch.zeros_like(self.actor_mu(torch.zeros_like(features))).to(device)
                sigma = torch.ones_like(mu).to(device) * 0.1
                dist = Normal(mu, sigma)
                return dist, value
        
        dist = Normal(mu, sigma)  # 创建正态分布
    
        # Critic: 输出状态值
        with torch.cuda.amp.autocast(enabled=True):
            value = self.critic(features)
            
            # 数值检查
            if torch.isnan(value).any() or torch.isinf(value).any():
                print("警告:值中出现NaN或Inf，使用默认分布和值---------215")
                mu = torch.zeros_like(self.actor_mu(torch.zeros_like(features))).to(device)
                sigma = torch.ones_like(mu).to(device) * 0.1
                dist = Normal(mu, sigma)
                value = torch.zeros(1).to(device)
                return dist, value
        
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("警告:值中出现NaN或Inf，使用默认分布和值---------223")
            mu = torch.zeros_like(self.actor_mu(torch.zeros_like(features))).to(device)
            sigma = torch.ones_like(mu).to(device) * 0.1
            dist = Normal(mu, sigma)
            value = torch.zeros(1).to(device)
            return dist, value


        return dist, value

class PPO2:
    def __init__(self, node_num, env_information=None):
        """
        PPO算法，专门为抬腿训练调整
        Args:
            node_num: 图节点数量
            env_information: 环境信息
        """
        self.node_num = node_num
        self.env_information = env_information
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler()
        
        # PPO超参数
        self.gamma = 0.99  # 折扣因子
        self.gae_lambda = 0.95  # GAE参数
        self.clip_ratio = 0.2  # PPO裁剪参数
        self.value_coef = 0.5  # 值函数损失系数
        self.entropy_coef = 0.02  # 熵正则化系数，对于抬腿训练略微增大
        self.max_grad_norm = 0.5  # 梯度裁剪阈值，减小以防止梯度爆炸
        
        # 学习率和优化器参数
        self.lr = 2e-4
        self.lr_decay = 0.995  # 学习率衰减
        self.warmup_steps = 2000  # 学习率预热步数
        self.current_step = 0  # 当前训练步数
        
        # PPO更新参数
        self.policy_update_epochs = 5  # 减少更新轮数以加快训练
        self.batch_size = 32  # 减小批量大小
        self.mini_batch_size = 8  # 小批量大小

        # 初始化策略网络
        self.policy = ActorCritic(act_dim=1, node_num=self.node_num).to(device) 
        
        # 使用Adam优化器，添加权重衰减
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), 
            lr=self.lr,
            weight_decay=1e-4  # 添加权重衰减
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10, 
            min_lr=1e-6,
            verbose=True
        )
        
        # 存储轨迹数据
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # 记录训练统计信息
        self.episode_rewards = []
        self.best_reward = -float('inf')
    
    def choose_action(self, episode_num, obs, x_graph):
        """
        选择动作
        Args:
            episode_num: 当前回合数
            obs: 观察数据
            x_graph: 图特征数据
        Returns:
            动作、对数概率和状态值
        """
        with torch.no_grad():
            dist, value = self.policy(x=obs[0], state=obs[1], x_graph=x_graph)
            
            if dist is None:
                print("Warning: dist is None, using random action")
                # 返回一个随机动作
                action = torch.tensor([np.random.uniform(-1.0, 1.0)], device=device)
                log_prob = torch.tensor(0.0, device=device)
                value = torch.tensor(0.0, device=device)
            else:
            
                action = dist.sample()
                # # 抬腿训练增加一些探索
                # if episode_num < 500:  # 前500回合增加探索
                #     action = dist.sample()
                # else:
                #     # 减少探索，更倾向于选取高概率动作
                #     if np.random.random() < 0.8:
                #         action = torch.argmax(action_probs)
                #     else:
                #         action = dist.sample()
            action_clamped = torch.clamp(action, -1.0, 1.0)        
            log_prob = dist.log_prob(action).sum(dim=-1)
            # 将对数概率和状态值保留到6位小数
            log_prob_rounded = round(log_prob.item(), 6)
            value_rounded = round(value.item(), 6)
           
            return  action_clamped.item(), log_prob_rounded, value_rounded
    
    # def store_transition_catch(self, state, action_shoulder, action_arm, reward, next_state, done, value, log_prob):
    #     """存储轨迹数据"""
    #     self.states.append(state)
    #     self.action_shoulder.append(action_shoulder)
    #     self.action_arm.append(action_arm)
    #     self.rewards.append(reward)
    #     self.next_states.append(next_state)
    #     self.values_shoulder.append(value)
    #     self.values_arm.append(value)
    #     self.log_probs_shoulder.append(log_prob)
    #     self.log_probs_arm.append(log_prob)
    #     self.dones.append(done)
    
    def store_transition_tai(self, state, actions, rewards, next_state, done, value, log_prob):
        """存储轨迹数据"""
        self.states.append(state)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_states.append(next_state)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def calculate_advantages(self):
        """计算GAE优势函数"""
        advantages = []
        gae = 0
        
        # 将Python列表转换为PyTorch张量
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device)
        values = torch.tensor(self.values, dtype=torch.float32).to(device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(device)
        
        # 计算最后一个状态的值
        next_value = torch.tensor(0.0, device=device)
        
        # 反向计算GAE
        for i in reversed(range(len(rewards))):
            if i < len(rewards) - 1:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return torch.stack(advantages) if advantages else torch.tensor([], device=device)
    
    def learn(self):
        """PPO学习过程"""
        try:
            # 检查是否有足够的数据
            if len(self.states) < self.mini_batch_size:
                print('数据不足，跳过学习')
                return 0
                
            # 学习率预热
            if self.current_step < self.warmup_steps:
                lr_scale = min(1.0, float(self.current_step) / self.warmup_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr * lr_scale
        
            self.current_step += 1
                
            # 计算优势函数和回报
            with torch.cuda.amp.autocast():
                # 确保所有张量都在正确的设备上
                rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
                values = torch.tensor(self.values, dtype=torch.float32, device=device)
                dones = torch.tensor(self.dones, dtype=torch.float32, device=device)

                # 计算GAE
                advantages = []
                gae = 0
                next_value = 0.0  # 假设最后一个状态的价值为0

                for i in reversed(range(len(rewards))):
                    if i < len(rewards) - 1:
                        next_value = values[i + 1]
                    delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
                    gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
                    advantages.insert(0, gae)

                advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
                returns = advantages + values

                # 归一化优势函数
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # 转换为张量
                batch_actions = torch.tensor(self.actions, dtype=torch.float32, device=device)
                batch_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)

            # PPO更新
            total_loss = 0
            for _ in range(self.policy_update_epochs):
                # 重新评估动作和值
                all_means = []
                all_log_stds = []
                all_values = []
                valid_indices = []

                for i in range(len(self.states)):
                    try:
                        dist, value = self.policy(
                            x=self.states[i][0],
                            state=self.states[i][1],
                            x_graph=self.states[i][2]
                        )
                        if dist is not None:
                            valid_indices.append(i)
                            all_means.append(dist.mean)
                            all_log_stds.append(dist.scale.log())
                            all_values.append(value)
                    except Exception as e:
                        print(f"Error in forward pass for sample {i}: {e}")
                        continue

                if not valid_indices:
                    print("Warning: No valid samples in this batch, skipping update")
                    continue

                # 只处理有效样本
                batch_states = [self.states[i] for i in valid_indices]
                batch_actions = batch_actions[valid_indices]
                batch_log_probs = batch_log_probs[valid_indices]
                advantages_batch = advantages[valid_indices]
                returns_batch = returns[valid_indices]

                all_means = torch.stack(all_means)
                all_log_stds = torch.stack(all_log_stds)
                all_values = torch.cat(all_values)

                # 确保所有张量都需要梯度
                all_means.requires_grad_(True)
                all_log_stds.requires_grad_(True)
                all_values.requires_grad_(True)

                with torch.cuda.amp.autocast():
                    # 创建分布
                    dist = torch.distributions.Normal(all_means, torch.exp(all_log_stds))
                    
                    # 计算新的对数概率
                    new_log_probs = dist.log_prob(batch_actions.unsqueeze(-1))
                    
                    # 计算熵
                    entropy = dist.entropy().mean()
                    
                    # 计算比率
                    ratio = torch.exp(new_log_probs - batch_log_probs.unsqueeze(-1))
                    
                    # 裁剪比率
                    surr1 = ratio * advantages_batch.unsqueeze(-1)
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_batch.unsqueeze(-1)
                    
                    # 策略损失
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # 值函数损失
                    value_loss = nn.MSELoss()(all_values.squeeze(), returns_batch)
                    
                    # 总损失
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                    # 检查最终损失
                    if torch.isnan(loss).any():
                        print("Warning: NaN in final loss")
                        print(f"policy_loss: {policy_loss.item()}, value_loss: {value_loss.item()}, entropy: {entropy.item()}")
                        print(f"loss: {loss.item()}")
                        continue
                
                # 梯度清零
                self.optimizer.zero_grad()
                
                # 使用混合精度训练
                with torch.cuda.amp.autocast():
                    # 缩放损失并反向传播
                    self.scaler.scale(loss).backward()
                    
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                    
                    # 更新参数
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                
                total_loss += loss.item()
            
            # 清空轨迹数据
            self.states = []
            self.actions = []
            self.rewards = []
            self.next_states = []
            self.values = []
            self.log_probs = []
            self.dones = []

            # 更新学习率
            self.scheduler.step(total_loss)
            
            return total_loss / self.policy_update_epochs if self.policy_update_epochs > 0 else 0
            
        except Exception as e:
            print(f"Error in learn method: {e}")
            import traceback
            traceback.print_exc()
            # 清空轨迹数据以防出错导致的数据累积
            self.states = []
            self.actions = []
            self.rewards = []
            self.next_states = []
            self.values = []
            self.log_probs = []
            self.dones = []
            return 0

