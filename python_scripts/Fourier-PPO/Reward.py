import numpy as np

# ===================== 奖励计算器 =====================
class RewardCalculator:
    """
    根据文档计算的奖励计算器
    """
    def __init__(self):
        self.prev_distance = None
        
    def compute_reward(self, gps_current, gps_goal, grasp_sensors, steps, done, 
                      success_flag1, current_distance=None):
        """
        计算奖励（基于文档的奖励机制）
        
        参数:
        - gps_current: 当前GPS坐标 [x, y]
        - gps_goal: 目标GPS坐标 [x, y]
        - grasp_sensors: 抓取传感器值列表 [6个值]
        - steps: 当前步数
        - done: 是否结束
        - success_flag1: 抓取成功标志
        - current_distance: 当前距离（如果已计算）
        """
        # 计算当前距离（如果没有提供）
        if current_distance is None:
            dx = gps_goal[0] - gps_current[0]
            dy = gps_goal[1] - gps_current[1]
            current_distance = np.sqrt(dx**2 + dy**2)
        
        # 1. 高密度距离奖励
        if self.prev_distance is not None:
            distance_reward = (self.prev_distance - current_distance) * 10.0
        else:
            distance_reward = -current_distance
        
        self.prev_distance = current_distance
        
        total_reward = distance_reward
        
        # 2. 稀疏抓取奖励
        if success_flag1 == 1:  # 抓到了
            if current_distance <= 0.04:  # 4 cm 容忍
                total_reward += 100
                print("✅ 抓到目标梯级，发放大奖励！")
            else:
                total_reward -= 60  # 抓错梯子，无大奖励
                print("⚠️  抓到非目标梯级，无大奖励")
        
        # 3. 错误抓取惩罚
        if done == 1 and steps < 6 and success_flag1 != 1:
            print("错误抓取！给予较大惩罚！")
            total_reward -= 100
        
        if done == 1 and steps <= 2 and success_flag1 != 1:
            print("因环境不稳定导致无效数据，跳过此步骤！！！")
            # 返回一个特殊标志，表示跳过
            return None, True
        
        # 4. 步数惩罚（在训练循环中应用）
        # steps * 0.5 将在训练循环中扣除
        
        return total_reward, False
    
    def reset(self):
        """重置奖励计算器（在每个episode开始时调用）"""
        self.prev_distance = None
