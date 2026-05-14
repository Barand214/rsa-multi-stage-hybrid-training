import numpy as np

# ===================== 模拟环境（用于测试） =====================
class MockEnv:
    """模拟环境，用于测试算法"""
    def __init__(self, n_servos=20):
        self.n_servos = n_servos
        self.gps_goal = np.array([1.0, 1.0])  # 目标位置
        self.step_count = 0
        
    def reset(self):
        """重置环境"""
        self.step_count = 0
        
        # 随机生成初始状态
        image = np.random.rand(3, 480, 640).astype(np.float32)
        angles = np.random.uniform(-1, 1, self.n_servos).astype(np.float32)
        gps_current = np.array([0.0, 0.0])
        grasp_sensors = [0, 0, 0, 0, 0, 0]
        
        return image, angles, gps_current, self.gps_goal.copy(), grasp_sensors
    
    def step(self, action):
        """执行一步"""
        self.step_count += 1
        
        # 模拟状态变化
        next_image = np.random.rand(3, 480, 640).astype(np.float32)
        next_angles = action.copy()  # 假设动作就是角度
        
        # 模拟GPS变化（随机走动）
        next_gps = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)])
        
        # 模拟抓取传感器（随机触发）
        grasp_sensors = [np.random.choice([0, 1]) for _ in range(6)]
        
        # 模拟结束条件
        done = np.random.rand() < 0.01 or self.step_count >= 100  # 1%概率结束或最多100步
        
        info = {
            'grasp_sensors': grasp_sensors
        }
        
        return next_image, next_angles, next_gps, done, info