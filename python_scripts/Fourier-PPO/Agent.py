from Network import FourierActorNetwork, CriticNetwork
from FourierActionSpace import FourierActionSpace
from MultiModalStateEncoder import MultiModalStateEncoder
from Reward import RewardCalculator
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical


# ===================== PPO智能体 =====================
class PPOAgent:
    """
    PPO智能体主类
    """
    def __init__(self, config):
        self.config = config
        
        # 动作空间
        self.action_space = FourierActionSpace(
            n_servos=config['n_servos'],
            max_harmonics=config['max_harmonics'],
            T_max=config['T_max']
        )
        
        # 状态编码器
        self.state_encoder = MultiModalStateEncoder(
            image_shape=config['image_shape'],
            n_servos=config['n_servos'],
            hidden_dim=config['hidden_dim']
        )
        
        # Actor和Critic网络
        self.actor = FourierActorNetwork(
            state_dim=config['hidden_dim'],
            action_space=self.action_space,
            hidden_dim=config['hidden_dim']
        )
        self.critic = CriticNetwork(
            state_dim=config['hidden_dim'],
            hidden_dim=config['hidden_dim']
        )
        
        # 优化器
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config['actor_lr']
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config['critic_lr']
        )
        
        # 经验缓冲区
        self.buffer = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': [],
            'images': [],
            'angles': [],
            'time_steps': []
        }
        
        # 训练参数
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.ppo_epochs = config.get('ppo_epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
    def encode_state(self, image, angles, time_step):
        """编码状态"""
        # 转换为tensor
        image_tensor = torch.FloatTensor(image).unsqueeze(0)
        angles_tensor = torch.FloatTensor(angles).unsqueeze(0)
        time_tensor = torch.FloatTensor([time_step]).unsqueeze(0)
        
        # 编码
        with torch.no_grad():
            encoded_state = self.state_encoder(image_tensor, angles_tensor, time_tensor)
        
        return encoded_state.squeeze(0)
    
    def select_action(self, state, deterministic=False):
        """选择动作"""
        state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
        
        # Actor输出
        action_params, log_prob, dists = self.actor.sample(state_tensor, deterministic)
        
        # Critic价值估计
        with torch.no_grad():
            value = self.critic(state_tensor)
        
        # 转换为numpy
        action_params_np = {}
        for key, val in action_params.items():
            if isinstance(val, torch.Tensor):
                action_params_np[key] = val.detach().cpu().numpy()
            else:
                action_params_np[key] = val
        
        return action_params_np, log_prob, value.squeeze(-1), dists
    
    def store_transition(self, image, angles, time_step, action, log_prob, reward, value, done):
        """存储经验"""
        # 编码状态
        encoded_state = self.encode_state(image, angles, time_step)
        
        self.buffer['states'].append(encoded_state.detach().cpu().numpy())
        self.buffer['actions'].append(action)
        self.buffer['log_probs'].append(log_prob.detach().cpu().numpy() if log_prob is not None else 0)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value)
        self.buffer['dones'].append(done)
        self.buffer['images'].append(image)
        self.buffer['angles'].append(angles)
        self.buffer['time_steps'].append(time_step)
    
    def compute_advantages(self, rewards, values, dones, next_value):
        """计算GAE优势函数"""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        next_value = next_value.cpu().numpy() if isinstance(next_value, torch.Tensor) else next_value
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self):
        """PPO更新步骤"""
        if len(self.buffer['rewards']) < self.config['min_buffer_size']:
            return {}
        
        # 转换为numpy数组
        states = np.array(self.buffer['states'])
        actions = self.buffer['actions']
        old_log_probs = np.array(self.buffer['log_probs'])
        rewards = np.array(self.buffer['rewards'])
        values = np.array(self.buffer['values'])
        dones = np.array(self.buffer['dones'])
        
        # 计算最后一个状态的value
        last_image = self.buffer['images'][-1]
        last_angles = self.buffer['angles'][-1]
        last_time_step = self.buffer['time_steps'][-1]
        last_state = self.encode_state(last_image, last_angles, last_time_step).unsqueeze(0)
        with torch.no_grad():
            next_value = self.critic(last_state)
        
        # 计算优势函数和回报
        advantages, returns = self.compute_advantages(rewards, values, dones, next_value)
        
        # 转换为tensor
        states_tensor = torch.FloatTensor(states)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs)
        returns_tensor = torch.FloatTensor(returns)
        advantages_tensor = torch.FloatTensor(advantages)
        
        # 多轮PPO更新
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        
        indices = np.arange(len(states))
        
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                
                # 重新计算新策略的概率
                _, batch_new_log_probs, dists = self.actor.sample(batch_states, deterministic=False)
                
                # 计算价值
                batch_values = self.critic(batch_states).squeeze(-1)
                
                # 概率比
                ratio = torch.exp(batch_new_log_probs - batch_old_log_probs)
                
                # PPO裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 
                                   1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic损失
                critic_loss = F.mse_loss(batch_values, batch_returns)
                
                # 熵正则化
                entropy_loss = 0
                # 计算熵（简化版，只计算离散分布的熵）
                if 'n_logits' in dists:
                    n_dist = Categorical(logits=dists['n_logits'])
                    entropy_loss = -n_dist.entropy().mean()
                
                # 总损失
                total_loss = (actor_loss + 
                            self.value_coef * critic_loss + 
                            self.entropy_coef * entropy_loss)
                
                # 反向传播
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                # 更新参数
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # 记录损失
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss)
        
        # 清空缓冲区
        self.clear_buffer()
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy_loss': np.mean(entropy_losses)
        }
    
    def clear_buffer(self):
        """清空经验缓冲区"""
        for key in self.buffer:
            self.buffer[key] = []
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'state_encoder_state_dict': self.state_encoder.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.state_encoder.load_state_dict(checkpoint['state_encoder_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
