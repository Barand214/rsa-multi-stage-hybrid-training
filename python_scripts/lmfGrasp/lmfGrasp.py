import torch
import torch.nn as nn
import numpy as np
import sys
import math

from replay_memory import ReplayMemory as replayMemory
import torch.nn.functional as F
import torchvision.transforms as transforms
from DarwinOP2 import Darwin
from RobotControl import RobotRun

# GPU设置
# if  torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"

file1_path = 'E:/pati/static/train/DQN/'

# robot = Robot()

LR = 0.0001   # 学习率
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64   # 批处理的样本数量
GAMMA = 0.99   # 计算价值函数的折扣因子
TARGET_REPLACE_ITER = 100   # 目标网络的更新频率



class jie_duan1_Env():    # 子环境类
    def __init__(self):
        self.darwin_robot = Darwin()
        self.robot = self.darwin_robot.robot
        self.state = None
        self.done = False
        self.isSuccess = False
        self.sensorprocessor = SensorDataPreprocessor()

    def step(self, state, action, steps, zhua, gps1, gps2, gps3, gps4, name):    # 仿真平台更新函数
        return RobotRun(self.robot, state, action, steps, zhua, gps1, gps2, gps3, gps4, name).run()

    def reset(self):   # 重置状态函数
        self.robot_reset()   # 重置机器人状态
        with open(file1_path + 'resetFlag.txt', 'r+') as file:   # 以记事本的形式传参，作为标识符
            file.write('0')
        with open(file1_path + 'resetFlag1.txt', 'r+') as file:
            file.write('0')
        self.done = False
        return self.state

    def robot_reset(self):   # 重置机器人的舵机角度参数
        return self.darwin_robot.robot_reset()

    def print_gps(self):   # 获取机器人GPS当前的位置参数
        return self.darwin_robot.print_gps()

    def get_img(self, steps):   # 获取机器人摄像头观察到的图像信息
        return self.darwin_robot.get_img(steps)

    def get_position(self):   # 获取机器人的当前的舵机角度参数
        return self.darwin_robot.getPosition()

    def wait_reset(self, s):   # 暂时停止控制代码向下运行，为仿真环境中机器人舵机运动留出一定时间
        return self.darwin_robot.wait(s)

    def getState(self):
        joint_data = self.darwin_robot.getPosition()
        angle_data = self.darwin_robot.getAngle()
        gyro_data = self.darwin_robot.getAngularVelocity()
        get_state = self.sensorprocessor.process_fuse_state(joint_data, angle_data, gyro_data)

        return get_state



class SensorDataPreprocessor:
    def __init__(self):
        # 传感器校准参数（需要根据实际传感器调整）
        self.gyro_bias = 512.0  # 陀螺仪零偏
        self.gyro_scale = 2000.0 / 1024.0  # 转换为°/s

        # 关节限制（根据Darwin OP2实际情况调整）
        self.joint_limits = {
            'min': [-3.14, -3.14, -0.68, -2.25, -1.65, -1.18, -2.24, -0.69, -1.01, -1.0,-1.77, -0.5, -0.02, -2.25, -1.24, -1.39, -0.68, -1.02, -1.81, -0.36],
            'max': [3.14, 2.85, 2.3, 0.77, 1.16, 1.63, 0.66, 2.5, 1.01, 0.93, 0.45, 1.68, 2.25, 0.03, 1.38, 1.22, 1.04, 0.60, 1.81, 0.94]
        }

    def preprocess_gyro(self, gyro_raw):
        """
        预处理陀螺仪数据：校准和单位转换
        """
        gyro_calibrated = (np.array(gyro_raw) - self.gyro_bias) * self.gyro_scale
        # 转换为弧度/秒
        gyro_rad = np.radians(gyro_calibrated)
        return gyro_rad

    def preprocess_joints(self, joint_positions):
        """
        预处理关节位置数据
        """
        joints = np.array(joint_positions)

        # 1. 归一化到[-1, 1]范围
        joints_normalized = np.zeros_like(joints)
        for i in range(len(joints)):
            min_val = self.joint_limits['min'][i]
            max_val = self.joint_limits['max'][i]
            joints_normalized[i] = 2 * (joints[i] - min_val) / (max_val - min_val) - 1

        # 2. 处理异常值
        joints_normalized = np.clip(joints_normalized, -1.0, 1.0)

        return joints_normalized

    def preprocess_angles(self, angles):
        """
        预处理姿态角
        """
        # 角度已经在合理范围内，只需确保数值稳定性
        angles_array = np.array(angles)
        return np.clip(angles_array, -math.pi, math.pi)

    def process_fuse_state(self, joint_data, angle_data, gyro_data):
        """
        处理单帧数据
        """
        # 预处理各传感器数据
        joints_processed = self.preprocess_joints(joint_data)
        angles_processed = self.preprocess_angles(angle_data)
        gyro_processed = self.preprocess_gyro(gyro_data)


        # 拼接单帧状态, 获取状态，目前还缺关节速度、偏航角
        frame_state = np.concatenate([
            joints_processed,  # 20维
            angles_processed,  # 2维
            gyro_processed  # 3维
        ])

        return frame_state



class LMFModule(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, rank):
        super().__init__()
        self.rank = rank
        self.hidden_dim = hidden_dim

        self.fc_x_list = nn.ModuleList([
            nn.Linear(input_dim1, hidden_dim) for _ in range(rank)
        ])
        self.fc_s_list = nn.ModuleList([
            nn.Linear(input_dim2, hidden_dim) for _ in range(rank)
        ])

        self.fc_fusion = nn.Linear(rank * hidden_dim, hidden_dim)

    def forward(self, x, state):
        batch_size = x.size(0)
        fusion_tensor = torch.zeros(batch_size, self.rank, self.hidden_dim).to(x.device)

        for i in range(self.rank):
            x_proj = self.fc_x_list[i](x)
            s_proj = self.fc_s_list[i](state)
            fusion_tensor[:, i, :] = x_proj * s_proj  # Hadamard product

        fusion_flat = fusion_tensor.view(batch_size, -1)
        fused = self.fc_fusion(fusion_flat)
        return fused


class Net(nn.Module):  # 网络参数类
    def __init__(self, act_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=1)

        self.fc0 = nn.Linear(6272, 6000)
        self.fc1 = nn.Linear(6000, 100)
        self.fc2 = nn.Linear(25, 100)
        self.fc3 = nn.Linear(100, 100)

        # 替代 concat 融合的 LMF 模块
        self.lmf = LMFModule(input_dim1=100, input_dim2=100, hidden_dim=200, rank=5)

        self.fc4 = nn.Linear(200, act_dim)

    def forward(self, x, state):


        x = torch.tensor(x).to('cuda')
        x = torch.unsqueeze(x, dim=0).unsqueeze(0).float()  # [1, 1, 128, 128]

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten

        x = self.fc0(x)
        x = self.fc1(x)

        # print(f"state:{state},len_state:{len(state)}")

        state = torch.tensor(state).to('cuda').unsqueeze(0).float()  # [1, 25]
        # 检查状态维度是否正确
        if state.shape[1] != 25:
            raise ValueError(f"状态维度错误: 预期25维, 实际{state.shape[1]}维")

        state = self.fc2(state)
        state = self.fc3(state)

        # 使用 LMF 代替 concat
        fused = self.relu(self.lmf(x, state))

        output = self.fc4(fused)
        output = output.squeeze()

        # print(f"output:{output}")
        return output



class DQN(object):   # DQN算法类
    def __init__(self):
        # 创建评估网络和目标网络
        self.eval_net,self.target_net =Net(2).to('cuda'),Net(2).to('cuda')   # 初始化两个结构相同的神经网络，一个作为评估网络，一个作为目标网络

        self.learn_step_counter = 0   # 学习步数记录
        self.memory_counter = 0   # 记忆量计数
        self.memory = np.zeros((MEMORY_CAPACITY,6))   # 存储空间初始化，每一组的数据为(o_t,s_t,a_t,r_t,o_{t+1},s_{t+1})
        self.optimazer = torch.optim.Adam(self.eval_net.parameters(),lr=LR)   # 初始化优化器
        self.loss_func = nn.MSELoss()     # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.loss_func = self.loss_func.to()

    def choose_action(self, ci_shu, x, y):  # 定义动作选择函数 (x为图像，y为状态)
        # 此处通过设置阈值，来调节机器人随机选择动作的概率
        if ci_shu < 1:
            yu_zhi = 0
        elif ci_shu >= 1 and ci_shu < 1000:
            yu_zhi = 1
        elif ci_shu >= 1000 and ci_shu < 2000:
            yu_zhi = 1
        elif ci_shu >= 2000 and ci_shu < 3000:
            yu_zhi = 1
        elif ci_shu >= 3000 and ci_shu < 20000:
            yu_zhi = 1
        else:
            yu_zhi = 1
        if np.random.uniform() < yu_zhi:   # 生成一个在[0, 1)内的随机数，如果小于阈值，选择最优动作；如果大于阈值，选择随机动作
            actions_value = self.eval_net.forward(x, y)   # 通过对评估网络输入图像x和状态y，前向传播获得动作值
            new_actions_value = torch.unsqueeze(actions_value, dim=0)
            action = torch.max(new_actions_value, dim=1)   # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[1]   # 输出action
            # print(f"actions_value: {actions_value}, action: {action}")
        else:  # 随机选择动作
            action = np.random.randint(0, 2)   # 有几个动作组就设置多少
        return action

    def store_transition(self, o, s, a, r, o_, s_):   # 储存训练样本到经验样本回放池
        s = [s]
        s_ = [s_]
        transition = np.hstack((o, s, [a, r], o_, s_))  # 分别为当前摄像头获取到的图像、当前机器人的全身舵机（20个）角度参数、当前动作和获取到的奖励值、下一状态的摄像头获取到的图像和下一状态机器人的全身舵机（20个）角度参数
        print(f"transition1:{transition.shape}")

        index = self.memory_counter % MEMORY_CAPACITY   # 如果样本池满了，新样本会替换旧样本
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self, rpm):   # 神经网络参数训练函数
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:   # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())   # 将评估网络的参数赋给目标网络
            print("我必须立刻学习")
        self.learn_step_counter += 1   # 学习步数自加1
        b_o, b_s, b_a, b_r, b_o_, b_s_, done = rpm.sample(64)   # 从样本经验池随机选取64组样本用于训练

        loss_all = 0   # 总损失函数初始化
        for i in range(64):   # 通过对64组样本进行循环计算损失函数，最终获取总损失函数
            state = b_s[i]
            state_ = b_s_[i]
            if len(state) != 25 or len(state_) != 25:
                raise ValueError(f"样本状态维度错误: 预期25维, 实际state={len(state)}维, state_={len(state_)}维")

            # print(f"rpm_state:{state}, len:{len(state)}")
            q_eval = self.eval_net(b_o[i], state)[[int(b_a[i])]]
            q_next = self.target_net(b_o_[i], state_)
            q_target = b_r[i] + GAMMA * q_next.max(0)[0]
            loss = self.loss_func(q_eval, q_target)
            loss_all = loss_all + loss
        loss_all = loss_all / 64
        self.optimazer.zero_grad()   # 清空上一步的残余更新参数值
        loss_all.backward()   # 误差反向传播
        self.optimazer.step()   # 逐步的梯度优化
        return loss_all

def main():   # 主函数
    dqn = DQN()
    ci_shu = 0
    rpm = replayMemory(100000)
    gps_goal = [0.2, 0.165]   # 设置目标基准坐标点
    for i in range(10000):  # 设置训练总轮次
        env = jie_duan1_Env()   # 环境初始化
        env.reset()   # 环境复位
        env.wait_reset(500)
        imgs = []
        steps = 0
        return_all = 0
        obs, obs_tensor = env.get_img(steps)
        robot_state = env.getState()
        # robot_state = env.get_robot_state()
        # robot_imu = env.get_robot_imu()
        # robot_state = env.getState()
        # print(f"state:{state}")

        # print(f"obs:{obs.shape}, obs_tensor:{obs_tensor.shape}")
        # print(f"robot_state:{robot_state}, robot_state_size:{len(robot_state)}")
        print(f"<<<<<<<<<<<<<<<<<<<<<<<第{i+1}轮>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        while True:
            a = dqn.choose_action(ci_shu, obs, robot_state)   # 获取当前动作
            gps1, gps2, gps3, gps4 = env.print_gps()   # 获取当前GPS坐标参数
            if steps >= 29:   # 设置每轮训练的最大运行步数，超过运行步数强行终止机器人运动，夹爪立即关闭
                zhua = 1.0
            else:
                zhua = 0.0
            name = "img" + str(steps) + ".png"
            next_state, reward, done, good, goal, count = env.step(robot_state, a, steps, zhua, gps1, gps2, gps3, gps4, name)   # 根据当前状态信息，控制机器人进行仿真运动，获得下一状态参数和动作奖励
            if count == 1:   # 如果需要进一步计算，则根据基准点的相对距离分档计算奖励
                x1 = gps_goal[0] - gps1[1]   # 计算机器人夹爪基准点与目标梯级基准点在X轴上的的相对距离
                y1 = gps_goal[1] - gps1[2]   # 计算机器人夹爪基准点与目标梯级基准点在Y轴上的的相对距离
                ju_li = math.sqrt(x1 * x1 + y1 * y1)   # 由于机器人夹爪默认正对目标梯级，忽略Z轴上的相对距离，直接计算欧式距离
                if ju_li > 0.06:   # 根据距离远近按档计算奖励
                    reward1 = 0
                elif ju_li > 0.03:
                    reward1 = 0.5
                else:
                    reward1 = 2
                reward = reward1
            return_all = return_all + reward   # 计算当前轮次的总奖励
            steps += 1
            next_obs, next_obs_tensor = env.get_img(steps)
            if good == 1:   # 如果此次机器人运行正常，将合格的数据储存到样本经验回放池中
                rpm.append((obs, robot_state, a, reward, next_obs, next_state, done))
            robot_state = env.getState()

            print(f"steps:{steps}, good:{good}, zhua:{zhua}, done:{done}")

            print(f"---------------当前次数为：{ci_shu}--------------------")
            if len(rpm) < 2000:   # 为机器人样本经验回放池设置阈值，当累计样本数低于阈值时，不进行训练，只是收集储存数据
                ci_shu = 0
            if len(rpm) > 2000 and done == 1:   # 当累计样本高于阈值，并且当前轮次训练结束时，对评估网络参数进行训练
                loss = dqn.learn(rpm)
                loss = loss.item()
                if goal == 1:   # 当机器人夹爪成功抓取到目标梯级时，将当前网络参数储存到指定位置，用于测试使用
                    save_path = r'E:\pati\static\train\model\lmf/lmfModel_%s.ckpt' % ci_shu
                    torch.save(dqn.eval_net, save_path)

                    with open(r'E:\pati\static\train\251106\02-lmf-reward.txt', 'a') as file:
                        file.write(str(ci_shu) + " " + str(return_all) + "\n")
                        file.close()

                    with open(r'E:\pati\static\train\251106\02-lmf-loss.txt', 'a') as file:
                        file.write(str(ci_shu) + " " + str(loss) + "\n")
                        file.close()


                if ci_shu % 100 == 0:   # 每100次训练，更新目标网络参数，并将当前网络参数储存到指定位置，用于测试使用
                    save_path = r'E:\pati\static\train\model\lmf/lmfModel_%s.ckpt' % ci_shu
                    torch.save(dqn.eval_net, save_path)

                with open(r'E:\pati\static\train\251106\02-lmf-goal.txt', 'a') as file:   # 如果当前轮次机器人成功抓取到目标梯级，则记录1到指定路径文本文档中，失败则记录0
                    goal_str = str(goal)
                    file.write(goal_str)
                    file.write(",")
                    file.close()
            if zhua == 1.0 or done == 1:   # 当机器人尝试抓取或者抓取中断时，重置训练环境，使用文本文档进行通信
                with open(file1_path + 'resetFlag.txt',
                          'r+') as file:
                    file.write('0')
                env.wait_reset(100)   # 等待一段时间用于通信
                env.reset()   # 仿真环境初始化
                imgs = []
                steps = 0
                ci_shu = ci_shu + 1

                break



if __name__ == '__main__':
    print("_________")
    with open(file1_path + 'resetFlag.txt', 'r+') as file:
        file.write('1')
    main()