import torch

if  torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# DQN超参数
BATCH_SIZE = 64                                 # 样本数量
LR = 0.0001                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.99                                     # reward discount
TARGET_REPLACE_ITER = 100                        # 目标网络更新频率(固定不懂的Q网络)
MEMORY_CAPACITY = 100000

# 路径配置
path_list = {
    'resetFlag': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/resetFlag.txt',
    'resetFlag1': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/resetFlag1.txt',
    'memory_graph': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/GNN/memory/memory_graph.pkl',
    'photo_path_real': 'E:/webotsone/Multi-Stage_Hybrid_Training/controllers/Train_main/',
    'photo_path': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/photo',
     
    #DQN 
    'model_path_catch_DQN': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/DQN/checkpoint/catch',
    'model_path_tai_DQN': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/DQN/checkpoint/tai',
    'shu_ju_path_DQN':'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/DQN/shu_ju.txt',
    'gps_path_DQN':'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/DQN/gps.txt',
    'catch_log_path_DQN':'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/DQN/log/catch_log/',
    'tai_log_path_DQN':'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/DQN/log/tai_log/',

    #PPO
    'model_path_catch_PPO': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/PPO/checkpoint/catch',
    'model_path_tai_PPO': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/PPO/checkpoint/tai',
    'shu_ju_path_PPO':'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/PPO/shu_ju.txt',
    'gps_path_PPO':'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/PPO/gps.txt',
    'catch_log_path_PPO':'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/PPO/log/catch_log/',
    'tai_log_path_PPO':'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/PPO/log/tai_log/',

    #SAC
    'model_path_catch_SAC': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/SAC/checkpoint/catch',
    'model_path_tai_SAC': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/SAC/checkpoint/tai',
    'shu_ju_path_SAC':'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/SAC/shu_ju.txt',
    'gps_path_SAC':'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/SAC/gps.txt',
    'catch_log_path_SAC':'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/SAC/log/catch_log/',
    'tai_log_path_SAC':'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/SAC/log/tai_log/',

    #Diffusion Policy
    'model_path_catch_Diffusion Policy': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/Diffusion Policy/checkpoint/catch',
    'model_path_tai_Diffusion Policy': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/Diffusion Policy/checkpoint/tai',
    'shu_ju_path_Diffusion Policy':'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/Diffusion Policy/shu_ju.txt',
    'gps_path_Diffusion Policy':'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/Diffusion Policy/gps.txt',
    'catch_log_path_Diffusion Policy':'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/Diffusion Policy/log/catch_log/',
    'tai_log_path_Diffusion Policy':'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/Diffusion Policy/log/tai_log/',

    #DiffWave
    'model_path_catch_DiffWave': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/DiffWave/checkpoint/catch',
    'model_path_tai_DiffWave': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/DiffWave/checkpoint/tai',
    'shu_ju_path_DiffWave': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/DiffWave/shu_ju.txt',
    'catch_log_path_DiffWave': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/DiffWave/log/catch_log/',
    'tai_log_path_DiffWave': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/DiffWave/log/tai_log/',
    'diffwave_entry': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/DiffWave/DiffWave_episoid_1.py',
    'diffwave_tai_entry': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/DiffWave/DiffWave_episoid_2_1.py',
    'diffwave_policy_file': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/DiffWave/DiffWave_policy.py',
    'diffwave_policy2_file': 'E:/webotsone/Multi-Stage_Hybrid_Training/python_scripts/DiffWave/DiffWave_policy_2.py',

}


gps_goal = [0.2, 0.175]  # 目标位置坐标1
gps_goal1 = [0.27, 0.225]  # 目标位置坐标2
# SAC超参数
LR_ACTOR = 0.0001 # 策略网络学习率
LR_CRITIC = 0.0001 # 评论家网络学习率
TAU = 0.005 # 目标网络软更新参数
GAMMA = 0.99 # 折扣因子
ALPHA = 0.2 # 初始温度参数
AUTO_ENTROPY = True # 是否自动调整温度参数
HIDDEN_SIZE = 256 # 隐藏层大小
MEMORY_SIZE = 1000000 # 经验回放缓冲区大小
WARMUP_STEPS = 1000 # 预热步数
UPDATE_AFTER_STEP = 1000 # 多少步更新一次网络
UPDATE_EVERY = 50 # 每多少步更新一次目标网络
POLICY_UPDATE = 2 # 策略网络更新频率
# 机器人运动标准参数
class Darwin_config:
    # 添加最小高度和最小前进位置参数
    min_height = 0.2  # 夹爪最小高度
    min_forward = 0.18  # 夹爪最小前进位置
    gps_goal = [0.2, 0.175]  # 目标位置（高度，前进）
    standard_angle = -32.64359902756043  # 标准角度
    limit = [[-3.14, 3.14], [-3.14, 2.85], [-0.68, 2.30], [-2.25, 0.77], 
             [-1.65, 1.16], [-1.18, 1.63], [-2.42, 0.66], [-0.69, 2.50], 
             [-1.01, 1.01], [-1.00, 0.93], [-1.77, 0.45], [-0.50, 1.68],
             [-0.02, 2.25], [-2.25, 0.03], [-1.24, 1.38], [-1.39, 1.22], 
             [-0.68, 1.04], [-1.02, 0.60], [-1.81, 1.81], [-0.36, 0.94]]  # 限制
    touch_T = [1.0, 1.0]  # 压力传感器目标值
    touch_F = [0.0, 0.0]  # 压力传感器失败值
    acc_low = [480, 450, 580]  # 加速度传感器低值
    acc_high = [560, 530, 700]  # 加速度传感器高值
    gyro_low = [500, 500, 500]  # 陀螺仪低值
    gyro_high = [520, 520, 520]  # 陀螺仪高值





