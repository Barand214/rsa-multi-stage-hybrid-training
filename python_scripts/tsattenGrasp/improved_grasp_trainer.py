"""
改进的抓取训练主文件
基于原有的tsattenGrasp.py，集成以下改进：
1. 改进的奖励函数
2. 更好的训练配置
3. 增强的监控和日志
4. 自适应学习策略
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import sys
import time

# 导入改进的模块
from improved_reward import ImprovedRewardCalculator
from improved_training_config import ImprovedTrainingConfig, AdaptiveLearning, TrainingMonitor, create_optimizer
from replay_memory import ReplayMemory
from tsattenGrasp import DQN, Net  # 使用原有的网络架构（暂时保持）

# 导入原有模块
from RobotRun1 import RobotRun, Robot_env


class ImprovedGraspTrainer:
    """改进的抓取训练器"""
    
    def __init__(self, config=None):
        self.config = config or ImprovedTrainingConfig()
        self.config.print_config()
        
        # 初始化组件
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  使用设备: {self.device}")
        
        # 初始化网络
        self.dqn = DQN()
        
        # 初始化训练组件
        self.reward_calculator = ImprovedRewardCalculator()
        self.adaptive_learning = AdaptiveLearning(self.config)
        self.monitor = TrainingMonitor("improved_training_log.txt")
        self.rpm = ReplayMemory(self.config.MEMORY_CAPACITY)
        
        # 创建优化器
        self.optimizer = create_optimizer(self.dqn.eval_net, self.config)
        
        # 训练统计
        self.total_episodes = 0
        self.best_success_rate = 0.0
        self.last_save_episode = 0
        
        print("✅ 改进的抓取训练器初始化完成")
    
    def train_episode(self, env, episode):
        """训练单个episode"""
        # 重置环境和奖励计算器
        env.wait_reset(500)
        self.reward_calculator.reset()
        
        # 获取初始状态
        steps = 0
        total_reward = 0
        obs, obs_tensor = env.get_img(steps)
        robot_state = env.get_robot_state()
        
        episode_success = False
        episode_losses = []
        
        print(f"\n🚀 Episode {episode} 开始训练")
        
        while steps < self.config.MAX_EPISODE_STEPS:
            # 选择动作（使用自适应epsilon）
            epsilon = self.adaptive_learning.update_epsilon(episode)
            action = self.dqn.choose_action(episode, obs, robot_state)
            
            # 获取GPS坐标
            gps1, gps2, gps3, gps4 = env.print_gps()
            
            # 确定是否抓取（最后几步强制抓取）
            if steps >= self.config.MAX_EPISODE_STEPS - 1:
                grasp = 1.0
            else:
                grasp = 0.0
            
            # 执行动作
            name = f"img{steps}.png"
            next_state, old_reward, done, good, goal_reached, count = env.step(
                robot_state, action, steps, grasp, gps1, gps2, gps3, gps4, name
            )
            
            # 使用改进的奖励计算
            touch_values = env.get_touch_values() if hasattr(env, 'get_touch_values') else [0, 0]
            improved_reward = self.reward_calculator.compute_reward(
                gps1, touch_values, steps, action, done, goal_reached
            )
            
            total_reward += improved_reward
            
            # 获取下一状态
            next_obs, next_obs_tensor = env.get_img(steps + 1)
            next_robot_state = env.get_robot_state()
            
            # 存储经验
            if good == 1:
                self.rpm.append((obs, robot_state, action, improved_reward, next_obs, next_robot_state, done))
            
            # 学习（更频繁的学习）
            loss = None
            if (len(self.rpm) >= self.config.MIN_MEMORY_SIZE and 
                steps % self.config.LEARN_INTERVAL == 0):
                
                loss = self._learn_step()
                if loss is not None:
                    episode_losses.append(loss)
            
            # 更新状态
            obs = next_obs
            robot_state = next_robot_state
            steps += 1
            
            # 成功检查
            if goal_reached == 1:
                episode_success = True
                print(f"🎉 Episode {episode} 在步骤 {steps} 成功!")
                break
            
            # 打印详细信息（每5步）
            if steps % 5 == 0:
                print(f"  Step {steps:2d}: action={action}, reward={improved_reward:6.2f}, "
                      f"total={total_reward:6.2f}, loss={loss:.4f if loss else 'N/A'}")
        
        # Episode结束统计
        avg_loss = np.mean(episode_losses) if episode_losses else None
        
        # 更新成功率统计
        success_rate = self.adaptive_learning.update_success_stats(episode_success)
        
        # 自适应调整学习率
        if episode % 50 == 0:
            self.adaptive_learning.update_learning_rate(self.optimizer, success_rate)
        
        # 记录日志
        self.monitor.log_episode(episode, total_reward, avg_loss, episode_success, epsilon, 
                               self.adaptive_learning.current_lr)
        
        # 保存模型
        if self._should_save_model(episode, success_rate):
            self._save_model(episode, success_rate)
        
        return episode_success, total_reward, avg_loss, success_rate
    
    def _learn_step(self):
        """执行一步学习"""
        try:
            loss = self.dqn.learn(self.rpm)
            return loss.item() if hasattr(loss, 'item') else loss
        except Exception as e:
            print(f"⚠️  学习步骤出错: {e}")
            return None
    
    def _should_save_model(self, episode, success_rate):
        """判断是否应该保存模型"""
        # 定期保存
        if episode % self.config.SAVE_INTERVAL == 0:
            return True
        
        # 成功率提升时保存
        if success_rate > self.best_success_rate + 0.05:  # 提升5%以上
            self.best_success_rate = success_rate
            return True
        
        # 达到里程碑时保存
        milestones = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        if any(success_rate >= milestone and self.best_success_rate < milestone for milestone in milestones):
            return True
        
        return False
    
    def _save_model(self, episode, success_rate):
        """保存模型"""
        os.makedirs("checkpoint/improved", exist_ok=True)
        
        # 保存当前最好模型
        save_path = f'checkpoint/improved/best_model_ep{episode}_sr{success_rate:.3f}.ckpt'
        torch.save({
            'model_state_dict': self.dqn.eval_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': episode,
            'success_rate': success_rate,
            'config': self.config.get_config_dict(),
        }, save_path)
        
        print(f"💾 模型已保存: {save_path}")
        self.last_save_episode = episode
    
    def train(self, num_episodes=5000, start_episode=0):
        """主训练循环"""
        print(f"🎯 开始训练 {num_episodes} 个episodes")
        
        # 初始化环境
        env = Robot_env()
        
        # 训练统计
        episode_rewards = []
        episode_success_rates = []
        
        try:
            for episode in range(start_episode, start_episode + num_episodes):
                self.total_episodes = episode
                
                # 训练单个episode
                success, total_reward, avg_loss, success_rate = self.train_episode(env, episode)
                
                episode_rewards.append(total_reward)
                episode_success_rates.append(success_rate)
                
                # 每100个episode显示详细统计
                if episode % 100 == 0:
                    self._print_statistics(episode, episode_rewards, episode_success_rates)
                
                # 早停检查
                if success_rate >= 0.8:  # 达到80%成功率
                    print(f"🏆 训练完成! 在Episode {episode} 达到80%成功率!")
                    break
                
        except KeyboardInterrupt:
            print("\n⏹️  训练被用户中断")
        except Exception as e:
            print(f"\n❌ 训练过程出错: {e}")
        finally:
            # 最终保存
            self._save_model(self.total_episodes, episode_success_rates[-1] if episode_success_rates else 0)
            self._plot_training_curves(episode_rewards, episode_success_rates)
            
            print("\n🏁 训练结束")
    
    def _print_statistics(self, episode, rewards, success_rates):
        """打印详细统计信息"""
        recent_rewards = rewards[-100:]
        recent_success_rates = success_rates[-100:]
        
        stats = self.monitor.get_statistics()
        
        print(f"\n📊 Episode {episode} 统计:")
        print(f"   最近100个episodes平均奖励: {np.mean(recent_rewards):.2f}")
        print(f"   最近100个episodes成功率: {np.mean(recent_success_rates):.3f}")
        print(f"   历史最高成功率: {self.best_success_rate:.3f}")
        print(f"   当前学习率: {self.adaptive_learning.current_lr:.2e}")
        print(f"   当前探索率: {self.adaptive_learning.current_epsilon:.3f}")
        print(f"   经验池大小: {len(self.rpm)}")
    
    def _plot_training_curves(self, rewards, success_rates):
        """绘制训练曲线"""
        try:
            plt.figure(figsize=(15, 5))
            
            # 奖励曲线
            plt.subplot(1, 3, 1)
            plt.plot(rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.grid(True)
            
            # 成功率曲线
            plt.subplot(1, 3, 2)
            plt.plot(success_rates)
            plt.title('Success Rate')
            plt.xlabel('Episode')
            plt.ylabel('Success Rate')
            plt.grid(True)
            
            # 移动平均
            plt.subplot(1, 3, 3)
            window = 50
            if len(rewards) >= window:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                plt.plot(moving_avg)
                plt.title(f'Moving Average Reward (window={window})')
                plt.xlabel('Episode')
                plt.ylabel('Average Reward')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('checkpoint/improved/training_curves.png', dpi=300, bbox_inches='tight')
            print("📈 训练曲线已保存: checkpoint/improved/training_curves.png")
            
        except Exception as e:
            print(f"⚠️  绘制训练曲线失败: {e}")


def main():
    """主函数"""
    print("🤖 改进的抓取训练程序")
    print("=" * 50)
    
    # 创建配置
    config = ImprovedTrainingConfig()
    
    # 创建训练器
    trainer = ImprovedGraspTrainer(config)
    
    # 开始训练
    trainer.train(num_episodes=3000)  # 训练3000个episodes
    
    print("✅ 程序执行完成")


if __name__ == "__main__":
    main()