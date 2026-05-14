import numpy as np
from FourierActionSpace import FourierActionSpace
from MultiModalStateEncoder import MultiModalStateEncoder
from Agent import PPOAgent
from Reward import RewardCalculator
from Environment import MockEnv

# ===================== 配置参数 =====================
config = {
    # 环境参数
    'n_servos': 20,
    'image_shape': (3, 480, 640),  # (channels, height, width)
    'max_harmonics': 20,
    'T_max': 10.0,
    'step_interval': 0.1,  # 步长时间间隔(s)
    
    # 网络参数
    'hidden_dim': 512,
    'actor_lr': 3e-4,
    'critic_lr': 1e-3,
    
    # PPO参数
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'ppo_epochs': 10,
    'batch_size': 64,
    'min_buffer_size': 2048,
    'max_grad_norm': 0.5,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    
    # 训练参数
    'total_episodes': 10000,
    'log_interval': 10,
    'save_interval': 100,
}

# ===================== 主训练循环 =====================
def train_ppo(config):
    """PPO训练主函数"""
    
    # 初始化智能体
    agent = PPOAgent(config)
    
    # 初始化奖励计算器
    reward_calculator = RewardCalculator()
    
    # 训练统计
    stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'actor_losses': [],
        'critic_losses': [],
        'success_rate': []
    }
    
    # 训练循环
    for episode in range(config['total_episodes']):
        # 重置环境（这里需要您实现环境接口）
        # 假设环境返回: image, angles, gps_current, gps_goal, grasp_sensors
        image, angles, gps_current, gps_goal, grasp_sensors = env.reset()
        
        # 重置奖励计算器
        reward_calculator.reset()
        
        # 生成动作参数（整个episode的动作曲线）
        state = agent.encode_state(image, angles, 0)
        action_params, log_prob, value, _ = agent.select_action(state, deterministic=False)
        
        # 获取动作组总时长T
        T = action_params['T'].item() if isinstance(action_params['T'], np.ndarray) else action_params['T']
        
        # 计算总步数
        total_steps = int(T / config['step_interval'])
        
        # Episode变量
        episode_reward = 0
        episode_steps = 0
        success = False
        skip_episode = False
        
        # 执行动作组
        for step in range(total_steps):
            current_time = step * config['step_interval']
            
            # 1. 计算当前舵机角度
            angles_current = agent.action_space.get_fourier_curve(action_params, current_time)
            
            # 2. 执行动作（需要您实现环境接口）
            # 假设环境返回: next_image, next_angles, next_gps, done, info
            # info包含grasp_sensors等
            next_image, next_angles, next_gps, done, info = env.step(angles_current)
            
            # 3. 计算奖励
            # 获取抓取传感器值
            grasp_sensors_values = info.get('grasp_sensors', [0, 0, 0, 0, 0, 0])
            
            # 判断抓取成功
            left_sensors = grasp_sensors_values[0:3]
            right_sensors = grasp_sensors_values[3:6]
            left_any = any(left_sensors)
            right_any = any(right_sensors)
            success_flag1 = 1 if (left_any and right_any) else 0
            
            # 计算距离
            dx = gps_goal[0] - next_gps[0]
            dy = gps_goal[1] - next_gps[1]
            current_distance = np.sqrt(dx**2 + dy**2)
            
            # 计算奖励
            reward, should_skip = reward_calculator.compute_reward(
                next_gps, gps_goal, grasp_sensors_values, 
                step, done, success_flag1, current_distance
            )
            
            if should_skip:
                skip_episode = True
                break
            
            # 4. 应用步数惩罚（根据文档）
            reward_with_penalty = reward - step * 0.5
            
            # 5. 编码下一个状态
            next_state = agent.encode_state(next_image, next_angles, current_time)
            
            # 6. 获取下一个状态的价值
            _, _, next_value, _ = agent.select_action(next_state, deterministic=True)
            
            # 7. 存储经验
            agent.store_transition(
                image=image,
                angles=angles,
                time_step=current_time,
                action=action_params,
                log_prob=log_prob,
                reward=reward_with_penalty,
                value=value,
                done=done
            )
            
            # 8. 更新统计
            episode_reward += reward_with_penalty
            episode_steps += 1
            
            # 9. 检查是否成功
            if success_flag1 == 1 and current_distance <= 0.04:
                success = True
            
            # 10. 准备下一步
            image = next_image
            angles = next_angles
            gps_current = next_gps
            
            if done:
                break
        
        # 跳过无效episode
        if skip_episode:
            print(f"Episode {episode}: 跳过无效数据")
            continue
        
        # 应用最终的步数惩罚（如果还有）
        final_step_penalty = episode_steps * 0.5
        episode_reward -= final_step_penalty
        
        # 更新智能体
        if len(agent.buffer['rewards']) >= config['min_buffer_size']:
            losses = agent.update()
            
            if losses:
                stats['actor_losses'].append(losses['actor_loss'])
                stats['critic_losses'].append(losses['critic_loss'])
        
        # 记录统计
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_steps)
        stats['success_rate'].append(1 if success else 0)
        
        # 打印进度
        if (episode + 1) % config['log_interval'] == 0:
            avg_reward = np.mean(stats['episode_rewards'][-config['log_interval']:])
            avg_length = np.mean(stats['episode_lengths'][-config['log_interval']:])
            success_rate = np.mean(stats['success_rate'][-config['log_interval']:]) * 100
            
            print(f"Episode {episode + 1}/{config['total_episodes']}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Success Rate: {success_rate:.1f}%")
            
            if losses:
                print(f"  Actor Loss: {losses['actor_loss']:.4f}")
                print(f"  Critic Loss: {losses['critic_loss']:.4f}")
            print("-" * 50)
        
        # 保存模型
        if (episode + 1) % config['save_interval'] == 0:
            agent.save_model(f"ppo_model_episode_{episode+1}.pth")
    
    return stats



# ===================== 使用示例 =====================
if __name__ == "__main__":
    print("初始化PPO智能体...")
    
    # 初始化环境（模拟环境）
    env = MockEnv(n_servos=config['n_servos'])
    
    # 训练（在真实环境中，您需要替换MockEnv为您的真实环境）
    print("开始训练...")
    try:
        stats = train_ppo(config)
        print("训练完成！")
    except KeyboardInterrupt:
        print("训练被中断")
    
    # 保存最终模型
    agent = PPOAgent(config)
    agent.save_model("ppo_model_final.pth")
    print("模型已保存到 ppo_model_final.pth")