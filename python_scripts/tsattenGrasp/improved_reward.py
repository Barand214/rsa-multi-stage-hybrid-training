"""
改进的奖励函数设计
解决原有奖励函数的问题：
1. 值域不稳定 [-180, 20]
2. 缺少过程奖励
3. 成功条件过于苛刻
"""
import math
import numpy as np

class ImprovedRewardCalculator:
    def __init__(self):
        self.goal_position = [0.195, 0.155]  # 目标位置
        self.success_threshold = 0.04  # 成功距离阈值（4cm，比原来5cm更宽容）
        self.max_distance = 0.2  # 最大有效距离
        self.prev_distance = None  # 上一步的距离
        
    def reset(self):
        """每个episode开始时重置"""
        self.prev_distance = None
    
    def compute_reward(self, gps1, touch_values, step, action, done, goal_reached):
        """
        计算改进的奖励函数
        Args:
            gps1: 当前夹爪位置 [x, y, z]
            touch_values: 触觉传感器值 [sensor1, sensor2]
            step: 当前步数
            action: 选择的动作
            done: 是否结束
            goal_reached: 是否达到目标
        Returns:
            reward: 归一化的奖励值 [0, 100]
        """
        total_reward = 0.0
        
        # 1. 距离奖励（密集奖励，引导机器人朝目标移动）
        current_distance = self._calculate_distance(gps1)
        
        if current_distance <= self.max_distance:
            # 距离越近，奖励越高（非线性，鼓励更接近）
            distance_reward = 20 * (1 - current_distance / self.max_distance) ** 2
            total_reward += distance_reward
            
            # 2. 进步奖励（朝正确方向移动的奖励）
            if self.prev_distance is not None:
                progress = self.prev_distance - current_distance
                if progress > 0:
                    total_reward += progress * 50  # 进步奖励
                else:
                    total_reward += progress * 10  # 轻微惩罚后退
        
        self.prev_distance = current_distance
        
        # 3. 触觉反馈奖励（中间奖励，鼓励接触）
        touch_sum = sum(touch_values) if touch_values else 0
        if touch_sum > 0:
            # 有接触就给奖励，不要求完美
            touch_reward = touch_sum * 15  # 每个接触点15分
            total_reward += touch_reward
            
            # 如果两个传感器都接触，额外奖励
            if touch_sum >= 1.8:  # 允许一些误差
                total_reward += 10
        
        # 4. 步数效率奖励/惩罚
        if step > 0:
            # 鼓励早期完成，但不要太严厉
            efficiency_penalty = step * 0.3
            total_reward -= efficiency_penalty
        
        # 5. 动作多样性奖励（防止重复动作）
        if hasattr(self, 'last_action') and self.last_action == action:
            total_reward -= 0.5  # 轻微惩罚重复动作
        self.last_action = action
        
        # 6. 成功奖励（大奖励）
        if goal_reached == 1:
            success_reward = 100
            total_reward += success_reward
            print(f"🎉 抓取成功! 距离: {current_distance:.4f}m, 总奖励: {total_reward:.2f}")
        
        # 7. 失败惩罚（适度）
        if done and goal_reached != 1 and step >= 29:
            # 超时失败
            total_reward -= 5
        
        # 8. 奖励归一化和裁剪
        total_reward = np.clip(total_reward, -10, 120)
        
        # 调试信息
        if step % 5 == 0:  # 每5步打印一次
            print(f"Step {step}: dist={current_distance:.4f}, touch={touch_sum:.1f}, reward={total_reward:.2f}")
        
        return total_reward
    
    def _calculate_distance(self, gps1):
        """计算当前位置到目标的欧氏距离"""
        if len(gps1) >= 3:
            x_diff = self.goal_position[0] - gps1[1]  # GPS格式: [?, x, y, z]
            y_diff = self.goal_position[1] - gps1[2]
            distance = math.sqrt(x_diff * x_diff + y_diff * y_diff)
        else:
            distance = self.max_distance  # 如果GPS数据无效，返回最大距离
        return distance
    
    def get_reward_info(self):
        """返回奖励设计的详细信息"""
        return {
            "distance_reward": "20 * (1 - dist/max_dist)^2, 最大20分",
            "progress_reward": "前进奖励50x, 后退惩罚10x",
            "touch_reward": "接触奖励15分/传感器",
            "efficiency_penalty": "步数惩罚0.3分/步",
            "success_reward": "成功奖励100分",
            "total_range": "[-10, 120]",
            "target_distance": f"{self.success_threshold}m"
        }


class LegacyRewardCalculator:
    """原有奖励函数，用于对比"""
    
    def __init__(self):
        self.goal_position = [0.195, 0.155]
    
    def compute_reward(self, gps1, touch_values, step, action, done, goal_reached):
        """原有的奖励计算方式"""
        x1 = self.goal_position[0] - gps1[1]
        y1 = self.goal_position[1] - gps1[2]
        distance = math.sqrt(x1 * x1 + y1 * y1)
        
        if distance > 0.06:
            reward = 0
        elif distance > 0.03:
            reward = 0.5
        else:
            reward = 2
            
        return reward


if __name__ == "__main__":
    # 测试奖励函数
    improved_calc = ImprovedRewardCalculator()
    legacy_calc = LegacyRewardCalculator()
    
    # 模拟一些测试情况
    test_cases = [
        # [gps1, touch, step, action, done, goal]
        [[0, 0.195, 0.155, 0], [1.0, 1.0], 10, 0, True, 1],  # 完美成功
        [[0, 0.18, 0.14, 0], [0.5, 0.5], 15, 1, False, 0],   # 接近但未成功
        [[0, 0.15, 0.12, 0], [0.0, 0.0], 20, 0, False, 0],   # 距离较远
        [[0, 0.10, 0.10, 0], [0.0, 0.0], 29, 1, True, 0],    # 超时失败
    ]
    
    print("奖励函数对比测试:")
    print("=" * 60)
    
    for i, (gps1, touch, step, action, done, goal) in enumerate(test_cases):
        improved_calc.reset()
        
        improved_reward = improved_calc.compute_reward(gps1, touch, step, action, done, goal)
        legacy_reward = legacy_calc.compute_reward(gps1, touch, step, action, done, goal)
        
        print(f"测试案例 {i+1}:")
        print(f"  GPS: {gps1[1:]}, Touch: {touch}, Step: {step}")
        print(f"  改进奖励: {improved_reward:.2f}")
        print(f"  原有奖励: {legacy_reward:.2f}")
        print(f"  改进幅度: {improved_reward - legacy_reward:.2f}")
        print()
    
    print("奖励设计信息:")
    for key, value in improved_calc.get_reward_info().items():
        print(f"  {key}: {value}")