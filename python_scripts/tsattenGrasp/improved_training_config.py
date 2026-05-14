"""
改进的训练超参数配置
解决原有训练策略的问题：
1. 学习率过高
2. 开始训练时机太晚
3. 更新频率太低
4. 批大小不合理
"""
import torch
import torch.nn as nn
import numpy as np

class ImprovedTrainingConfig:
    """改进的训练配置"""
    
    # DQN核心参数
    LEARNING_RATE = 5e-5  # 降低学习率，提高稳定性
    BATCH_SIZE = 32       # 减少批大小，提高训练频率
    GAMMA = 0.99          # 折扣因子保持不变
    EPSILON_INIT = 0.9    # 初始探索率
    EPSILON_MIN = 0.05    # 最小探索率
    EPSILON_DECAY = 0.995 # 探索率衰减
    
    # 经验回放参数
    MEMORY_CAPACITY = 50000  # 减少内存容量，提高样本新鲜度
    MIN_MEMORY_SIZE = 500    # 大幅降低开始训练的阈值
    
    # 网络更新参数
    TARGET_REPLACE_ITER = 200  # 目标网络更新频率
    LEARN_INTERVAL = 5         # 每5步学习一次（而非每episode一次）
    
    # 训练监控
    SAVE_INTERVAL = 500        # 每500次训练保存一次模型
    EVAL_INTERVAL = 100        # 每100次训练评估一次
    LOG_INTERVAL = 50          # 每50次训练记录一次日志
    
    # 奖励参数
    REWARD_SCALE = 1.0         # 奖励缩放系数
    MAX_EPISODE_STEPS = 30     # 最大步数保持不变
    
    # 网络架构参数
    HIDDEN_DIM = 256          # 隐藏层维度
    IMAGE_SIZE = (128, 128)   # 图像大小
    STATE_DIM = 20            # 状态维度
    ACTION_DIM = 2            # 动作维度（保持现有）
    
    @classmethod
    def get_config_dict(cls):
        """获取所有配置的字典形式"""
        config = {}
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                config[attr] = getattr(cls, attr)
        return config
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("🔧 改进的训练配置:")
        print("=" * 50)
        config = cls.get_config_dict()
        for key, value in config.items():
            if not key.startswith('get') and not key.startswith('print'):
                print(f"{key:20s}: {value}")
        print("=" * 50)


class LegacyTrainingConfig:
    """原有训练配置，用于对比"""
    
    LEARNING_RATE = 1e-4      # 原来的学习率（偏高）
    BATCH_SIZE = 64           # 原来的批大小（偏大）
    MIN_MEMORY_SIZE = 2000    # 原来的阈值（太高）
    LEARN_INTERVAL = "on_done"  # 只在episode结束时学习
    TARGET_REPLACE_ITER = 100   # 目标网络更新频率
    

class AdaptiveLearning:
    """自适应学习策略"""
    
    def __init__(self, config):
        self.config = config
        self.current_lr = config.LEARNING_RATE
        self.current_epsilon = config.EPSILON_INIT
        self.success_count = 0
        self.total_episodes = 0
        self.recent_success_rate = 0.0
        self.success_history = []
        
    def update_epsilon(self, episode):
        """更新探索率"""
        self.current_epsilon = max(
            self.config.EPSILON_MIN,
            self.current_epsilon * self.config.EPSILON_DECAY
        )
        return self.current_epsilon
    
    def update_learning_rate(self, optimizer, success_rate):
        """根据成功率自适应调整学习率"""
        self.recent_success_rate = success_rate
        
        # 如果成功率太低，略微提高学习率
        if success_rate < 0.1:
            new_lr = min(self.current_lr * 1.1, 1e-4)
        # 如果成功率合理，保持学习率
        elif 0.1 <= success_rate <= 0.4:
            new_lr = self.current_lr
        # 如果成功率较高，略微降低学习率以稳定
        else:
            new_lr = max(self.current_lr * 0.95, 1e-6)
        
        if abs(new_lr - self.current_lr) > 1e-7:
            self.current_lr = new_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"📊 学习率调整为: {new_lr:.2e} (成功率: {success_rate:.3f})")
    
    def should_save_model(self, episode):
        """判断是否应该保存模型"""
        return (episode % self.config.SAVE_INTERVAL == 0 or 
                self.recent_success_rate > 0.5)
    
    def update_success_stats(self, success):
        """更新成功统计"""
        self.success_history.append(int(success))
        if len(self.success_history) > 100:  # 只保留最近100次
            self.success_history.pop(0)
        
        self.recent_success_rate = np.mean(self.success_history)
        return self.recent_success_rate


class TrainingMonitor:
    """训练过程监控"""
    
    def __init__(self, log_file="training_log.txt"):
        self.log_file = log_file
        self.episode_count = 0
        self.loss_history = []
        self.reward_history = []
        self.success_history = []
        
    def log_episode(self, episode, reward, loss, success, epsilon, lr):
        """记录每个episode的信息"""
        self.episode_count += 1
        self.reward_history.append(reward)
        if loss is not None:
            self.loss_history.append(loss)
        self.success_history.append(int(success))
        
        # 每50个episode记录一次详细信息
        if episode % 50 == 0:
            avg_reward = np.mean(self.reward_history[-50:])
            avg_loss = np.mean(self.loss_history[-10:]) if self.loss_history[-10:] else 0
            success_rate = np.mean(self.success_history[-100:])
            
            log_msg = (f"Episode {episode:5d} | "
                      f"Avg_Reward: {avg_reward:7.2f} | "
                      f"Loss: {avg_loss:7.4f} | "
                      f"Success_Rate: {success_rate:5.3f} | "
                      f"Epsilon: {epsilon:5.3f} | "
                      f"LR: {lr:.2e}")
            
            print(log_msg)
            
            # 写入日志文件
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_msg + "\n")
    
    def get_statistics(self):
        """获取训练统计信息"""
        if not self.reward_history:
            return {}
        
        recent_rewards = self.reward_history[-100:]
        recent_success = self.success_history[-100:]
        recent_loss = self.loss_history[-50:] if self.loss_history else []
        
        stats = {
            "total_episodes": self.episode_count,
            "avg_reward_recent": np.mean(recent_rewards),
            "success_rate_recent": np.mean(recent_success),
            "avg_loss_recent": np.mean(recent_loss) if recent_loss else 0,
            "max_reward": max(self.reward_history),
            "min_reward": min(self.reward_history),
        }
        return stats


def create_optimizer(model, config):
    """创建优化器"""
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        eps=1e-4  # 添加数值稳定性
    )
    return optimizer


def create_lr_scheduler(optimizer, config):
    """创建学习率调度器"""
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # 监控成功率（越高越好）
        factor=0.8,
        patience=500,  # 500个episode无改善再降低学习率
        min_lr=1e-6
    )
    return scheduler


if __name__ == "__main__":
    # 测试配置
    print("配置对比测试:")
    print("\n原有配置:")
    legacy = LegacyTrainingConfig()
    for attr in ['LEARNING_RATE', 'BATCH_SIZE', 'MIN_MEMORY_SIZE']:
        print(f"  {attr}: {getattr(legacy, attr)}")
    
    print("\n改进配置:")
    ImprovedTrainingConfig.print_config()
    
    print("\n改进理由:")
    print("1. 学习率从1e-4降低到5e-5，提高训练稳定性")
    print("2. 批大小从64降低到32，提高更新频率")
    print("3. 开始训练阈值从2000降低到500，更早开始学习")
    print("4. 从每episode学习一次改为每5步学习一次")
    print("5. 添加自适应学习率和训练监控")