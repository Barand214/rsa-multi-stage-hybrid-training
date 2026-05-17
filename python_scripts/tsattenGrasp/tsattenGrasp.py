import torch
import torch.nn as nn
import numpy as np
from controller import Robot, Motor, Motion, LED, Camera, Gyro, Accelerometer, PositionSensor, GPS
from PIL import Image
import time
import sys
import math

from numpy import dtype

sys.path.append('D:/Dev/Tools/Webots/projects/robots/robotis/darwin-op/libraries/python37')
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager
from RobotRun1 import RobotRun
from replay_memory import ReplayMemory as replayMemory
import torch.nn.functional as F


# GPU设置
# if  torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"

file1_path = 'E:/pati/static/train/DQN/'

robot = Robot()

LR = 0.0001   # 学习率
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64   # 批处理的样本数量
GAMMA = 0.99   # 计算价值函数的折扣因子
TARGET_REPLACE_ITER = 100   # 目标网络的更新频率

class jie_duan1_Env():    # 子环境类
    def __init__(self):
        self.robot = robot
        self.state = None
        self.done = False
        self.isSuccess = False
        self.shooting = Shooting()

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
        return self.shooting.robot_reset()

    def print_gps(self):   # 获取机器人GPS当前的位置参数
        return self.shooting.print_gps()

    def get_img(self, steps):   # 获取机器人摄像头观察到的图像信息
        return self.shooting.get_img(steps)

    def get_robot_state(self):   # 获取机器人的当前的舵机角度参数
        return self.shooting.get_robot_state()

    def wait_reset(self, s):   # 暂时停止控制代码向下运行，为仿真环境中机器人舵机运动留出一定时间
        return self.shooting.wait(s)


class Shooting():   # 总环境类
    def __init__(self):
        self.timestep = int(robot.getBasicTimeStep())   # 初始化环境的最小仿真时间步长，具体数值为仿真环境中设置的大小
        self.gaitManager = RobotisOp2GaitManager(robot, 'config.ini')   # 初始化Robotis-OP2机器人的步态控制器，确定参数配置文件
        self.motionManager = RobotisOp2MotionManager(robot)
        self.gaitManager.setBalanceEnable(True)

        # --------------------------------启动传感器----------------------------------
        self.motors = []   # 机器人舵机名称列表初始化
        self.motors_sensors = []   # 机器人舵机传感器列表初始化
        self.motorName = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                          'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL', 'PelvR',
                          'PelvL', 'LegUpperR', 'LegUpperL', 'LegLowerR', 'LegLowerL',
                          'AnkleR', 'AnkleL', 'FootR', 'FootL', 'Neck', 'Head', 'GraspL', 'GraspR')   # 机器人全身舵机名称，原有的20个＋新加的2个
        self.eyeLed = robot.getDevice('EyeLed')   # 获取机器人眼部LED灯的设备对象
        self.headLed = robot.getDevice('HeadLed')   # 获取机器人头部LE灯的设备对象
        self.camera = robot.getDevice('Camera')   # 获取机器人摄像头的设备对象
        self.accelerometer = robot.getDevice('Accelerometer')   # 获取机器人加速度测量仪的设备对象
        self.gyro = robot.getDevice('Gyro')   # 获取机器人陀螺仪的设备对象
        self.left_gps1 = robot.getDevice('left_gps1')   # 获取机器人GPS的设备对象
        self.right_gps1 = robot.getDevice('right_gps1')
        self.left_gps2 = robot.getDevice('left_gps2')
        self.right_gps2 = robot.getDevice('right_gps2')
        self.left_gps1.enable(self.timestep)   # 激活机器人GPS，仿真时间步长为环境的最小仿真时间步长
        self.right_gps1.enable(self.timestep)
        self.left_gps2.enable(self.timestep)
        self.right_gps2.enable(self.timestep)
        self.camera.enable(self.timestep)   # 激活机器人摄像头，仿真时间步长为环境的最小仿真时间步长
        self.accelerometer.enable(self.timestep)   # 激活机器人加速度测量仪，仿真时间步长为环境的最小仿真时间步长
        self.gyro.enable(self.timestep)   # 激活机器人陀螺仪，仿真时间步长为环境的最小仿真时间步长
        for i in range(len(self.motorName)):   # 依次处理机器人的全身舵机
            self.motors.append(robot.getDevice(self.motorName[i]))   # 根据舵机名称获取舵机的设备对象，将其放入motors列表中
            sensorName = self.motorName[i]
            sensorName = sensorName + 'S'   # Webots中舵机传感器的名称是舵机名称+S
            self.position = robot.getDevice(sensorName)   # 根据舵机传感器名称获取舵机传感器的设备对象
            self.position.enable(self.timestep)   # 激活舵机传感器
            self.motors_sensors.append(self.position)   # 将激活后的舵机传感器放入motor_sensors列表

        # ---------------------------------启动结束-----------------------------------

    def myStep(self):   # 单步仿真函数
        robot.step(self.timestep)   # 每次调用该函数，仿真环境就进行1个最小时间步长的运行

    def wait(self, ms):   # 时间段仿真函数
        startTime = robot.getTime()   # 通过设置一段时间，使得控制程序在此处循环等待，而仿真环境则可以一直运行
        s = ms / 1000
        while s + startTime >= robot.getTime():
            self.myStep()

    def robot_reset(self):   # 机器人舵机角度复位函数

        a = -np.random.random() * 0.4 + 0.7   # 动态参数，通过随机数在一定范围内，随机化机器人肩膀处的Shoulder舵机的初始角度
        b = -np.random.random() * 0.4 + 1.0   # 动态参数，通过随机数在一定范围内，随机化机器人手肘处的ArmLower舵机的初始角度
        self.myStep()
        # 优先初始化手部夹爪的舵机角度参数
        self.motors[20].setPosition(1)  # left   # 通过直接调用软件自带的setPosition函数，可以直接控制舵机的设备对象，使其以默认速度转动到指定角度
        self.motors[21].setPosition(1)  # right)
        self.wait(200)   # 使代码等待一段时间，直到仿真环境机器人手部夹爪舵机结束运动
        self.gaitManager.stop()   # 暂停机器人步态控制器
        self.motors[18].setPosition(0.2)  # neck
        self.motors[19].setPosition(0)  # head
        self.motors[7].setPosition(0.0)  # PelvYL
        self.motors[9].setPosition(0.0)  # PelvL
        self.motors[11].setPosition(0.5)  # LegUpperL
        self.motors[13].setPosition(-0.2)  # LegLowerL
        self.motors[15].setPosition(0)  # AnkleL
        self.motors[17].setPosition(0.0)  # FootL
        self.motors[6].setPosition(0.0)  # PelvYR
        self.motors[8].setPosition(0.0)  # PelvR
        self.motors[10].setPosition(-0.5)  # LegUpperR
        self.motors[12].setPosition(0.2)  # LegLowerR
        self.motors[14].setPosition(0)  # AnkleR
        self.motors[16].setPosition(0.0)  # FootR
        self.motors[1].setPosition(a)  # ShoulderL
        self.motors[3].setPosition(0.67)  # ArmUpperL
        self.motors[5].setPosition(-b)  # ArmLowerL
        self.motors[0].setPosition(-a)  # ShoulderR
        self.motors[2].setPosition(-0.67)  # ArmUpperR
        self.motors[4].setPosition(b)  # ArmLowerR

        self.motors[20].setVelocity(1)  # left   # 通过直接调用软件自带的setVelocity函数，可以直接控制舵机的设备对象，修改其转动速度
        self.motors[21].setVelocity(1)
        self.motors[18].setVelocity(1)  # neck
        self.motors[19].setVelocity(1) # head
        self.motors[7].setVelocity(1) # PelvYL
        self.motors[9].setVelocity(1)  # PelvL
        self.motors[11].setVelocity(1)  # LegUpperL
        self.motors[13].setVelocity(1)  # LegLowerL
        self.motors[15].setVelocity(1) # AnkleL
        self.motors[17].setVelocity(1)  # FootL
        self.motors[6].setVelocity(1)  # PelvYR
        self.motors[8].setVelocity(1)  # PelvR
        self.motors[10].setVelocity(1)  # LegUpperR
        self.motors[12].setVelocity(1)  # LegLowerR
        self.motors[14].setVelocity(1)  # AnkleR
        self.motors[16].setVelocity(1)  # FootR
        self.motors[1].setVelocity(1)  # ShoulderL
        self.motors[3].setVelocity(1)  # ArmUpperL
        self.motors[5].setVelocity(1)  # ArmLowerL
        self.motors[0].setVelocity(1)  # ShoulderR
        self.motors[2].setVelocity(1)  # ArmUpperR
        self.motors[4].setVelocity(1)  # ArmLowerR
        # 代码暂停一段时间，等待仿真环境中机器人舵机运动结束
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()
        self.myStep()

    def print_gps(self):   # 获取当前GPS数值的函数
        gps_data1 = self.left_gps1.getValues()   # Webots2021.a版本使用.getValues()来获取设备对象的具体参数
        gps_data2 = self.right_gps1.getValues()
        gps_data3 = self.left_gps2.getValues()
        gps_data4 = self.right_gps2.getValues()
        return gps_data1, gps_data2, gps_data3, gps_data4   # 返回4个GPS数值

    def get_img(self, steps):   # 获取当前摄像头观察到图像的函数
        img = "img%s.png" % steps  # 为当前摄像头图像命名，将steps（当前训练轮次仿真步数）作为动态输入
        self.camera.saveImage("imgs/" + img, 100)  # 使用软件内置函数保存图像，100为图像质量，范围为0-100，默认路径为控制器根目录
        path = 'imgs/%s' % img
        img = Image.open(path)   # 根据路径名称打开所保存的图像
        img = img.convert('L')
        img = img.resize((128, 128))
        img = np.array(img)
        img = img / 255.0

        img_tensor = torch.tensor(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        img_tensor = img_tensor.unsqueeze(0)

        return img, img_tensor

    def get_robot_state(self):   # 获取当前机器人舵机角度参数
        self.robot_state = []   # 建立用于储存数据的列表
        for i in range(len(self.motorName) - 2):   # 通过循环的形式收集原有的20个舵机的角度参数
            position = self.motors_sensors[i].getValue()
            self.robot_state.append(position)
        return self.robot_state



class SpatioTemporalAttention(nn.Module):
    def __init__(self, x_dim, state_dim, hidden_dim=200):
        super().__init__()
        # Attention mechanism for spatial and temporal features
        self.query = nn.Linear(x_dim, hidden_dim)
        self.key = nn.Linear(state_dim, hidden_dim)
        self.value = nn.Linear(state_dim, hidden_dim)
        self.hidden_dim = hidden_dim

        # Projection layers
        self.proj = nn.Linear(hidden_dim + x_dim, hidden_dim)

    def forward(self, x, state):
        # x: features from CNN (spatial)
        # state: features from state processing (temporal)

        # Compute attention scores
        q = self.query(x.unsqueeze(0))  # [1, x_dim] -> [1, hidden_dim]
        k = self.key(state.unsqueeze(0))  # [1, state_dim] -> [1, hidden_dim]
        v = self.value(state.unsqueeze(0))  # [1, state_dim] -> [1, hidden_dim]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(0, 1)) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, v)

        # Combine with original features
        combined = torch.cat([x, attended_values.squeeze(0)], dim=-1)
        output = self.proj(combined)

        return output


class Net(nn.Module):
    def __init__(self, act_dim):
        super().__init__()
        # Convolutional layers for spatial features
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        self.relu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)

        # Fully connected layers
        self.fc0 = nn.Linear(in_features=6272, out_features=6000)
        self.fc1 = nn.Linear(in_features=6000, out_features=100)
        self.fc2 = nn.Linear(in_features=20, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=100)

        # Spatio-temporal attention fusion
        self.attention_fusion = SpatioTemporalAttention(x_dim=100, state_dim=100)

        # Final layers
        self.fc4 = nn.Linear(in_features=200, out_features=200)  # 128 comes from attention hidden_dim
        self.fc5 = nn.Linear(in_features=200, out_features=act_dim)

    def forward(self, x, state):
        # Process input x (spatial features)
        x = torch.tensor(x).to('cuda')
        x = torch.unsqueeze(x, dim=0)
        x = x.float()

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = torch.flatten(x)
        x = x.float()

        # Process through FC layers
        x = self.fc0(x)
        x = self.fc1(x)

        # Process state (temporal features)
        state = torch.tensor(state).to('cuda')
        state = state.float()
        state = self.fc2(state)
        state = self.fc3(state)

        # Normalize features
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        state = (state - state.min()) / (state.max() - state.min() + 1e-8)

        # Spatio-temporal attention fusion
        state_x = self.attention_fusion(x, state)

        # Final processing
        state_x = self.fc4(state_x)
        state_x = self.fc5(state_x)

        return state_x


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
        else:  # 随机选择动作
            action = np.random.randint(0, 2)   # 有几个动作组就设置多少
        return action

    def store_transition(self, o, s, a, r, o_, s_):   # 储存训练样本到经验样本回放池
        s = [s]
        s_ = [s_]
        transition = np.hstack((o, s, [a, r], o_, s_))  # 分别为当前摄像头获取到的图像、当前机器人的全身舵机（20个）角度参数、当前动作和获取到的奖励值、下一状态的摄像头获取到的图像和下一状态机器人的全身舵机（20个）角度参数
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
    for i in range(50000):  # 设置训练总轮次
        torch.cuda.empty_cache()  # 清除不必要的内存

        env = jie_duan1_Env()   # 环境初始化
        env.reset()   # 环境复位
        env.wait_reset(500)
        steps = 0
        return_all = 0
        obs, obs_tensor = env.get_img(steps)
        robot_state = env.get_robot_state()

        print(f"<<<<<<<<<<<<<<<<<<<<<<<第{i + 1}轮>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        while True:
            a = dqn.choose_action(ci_shu, obs, robot_state)  # 获取当前动作
            gps1, gps2, gps3, gps4 = env.print_gps()  # 获取当前GPS坐标参数
            if steps >= 29:  # 设置每轮训练的最大运行步数，超过运行步数强行终止机器人运动，夹爪立即关闭
                zhua = 1.0
            else:
                zhua = 0.0
            name = "img" + str(steps) + ".png"
            next_state, reward, done, good, goal, count = env.step(robot_state, a, steps, zhua, gps1, gps2, gps3, gps4, name)  # 根据当前状态信息，控制机器人进行仿真运动，获得下一状态参数和动作奖励
            if count == 1:  # 如果需要进一步计算，则根据基准点的相对距离分档计算奖励
                x1 = gps_goal[0] - gps1[1]  # 计算机器人夹爪基准点与目标梯级基准点在X轴上的的相对距离
                y1 = gps_goal[1] - gps1[2]  # 计算机器人夹爪基准点与目标梯级基准点在Y轴上的的相对距离
                ju_li = math.sqrt(x1 * x1 + y1 * y1)  # 由于机器人夹爪默认正对目标梯级，忽略Z轴上的相对距离，直接计算欧式距离
                if ju_li > 0.06:  # 根据距离远近按档计算奖励
                    reward1 = 0
                elif ju_li > 0.03:
                    reward1 = 0.5
                else:
                    reward1 = 2
                reward = reward1
            return_all = return_all + reward  # 计算当前轮次的总奖励
            steps += 1
            next_obs, next_obs_tensor = env.get_img(steps)
            robot_state = env.get_robot_state()

            print(f"------cishu:{ci_shu}, steps:{steps}, good:{good}, done:{done}, zhua:{zhua}, goal:{goal}, 奖励值为：{return_all}")

            if good == 1:  # 如果此次机器人运行正常，将合格的数据储存到样本经验回放池中
                rpm.append((obs, robot_state, a, reward, next_obs, next_state, done))

            # print(f"----len(rpm):{len(rpm)}")

            if len(rpm) < 2000:  # 为机器人样本经验回放池设置阈值，当累计样本数低于阈值时，不进行训练，只是收集储存数据
                ci_shu = 0
            if len(rpm) > 2000 and done == 1:  # 当累计样本高于阈值，并且当前轮次训练结束时，对评估网络参数进行训练
                loss = dqn.learn(rpm)
                loss = loss.item()

                if ci_shu % 100 == 0:  # 每100次训练，更新目标网络参数，并将当前网络参数储存到指定位置，用于测试使用
                    save_path = 'checkpoint/attModel_%s.ckpt' % ci_shu
                    torch.save(dqn.eval_net, save_path)

                if goal == 1:  # 当机器人夹爪成功抓取到目标梯级时，将当前网络参数储存到指定位置，用于测试使用
                    save_path = 'checkpoint/attModel_%s.ckpt' % ci_shu
                    torch.save(dqn.eval_net, save_path)

                    with open(r'E:\webots\MultiPati\controllers\tsattenGrasp\files\attRew.txt', 'a') as file:
                        file.write(str(ci_shu) + " " + str(return_all) + "\n")
                        file.close()

                    with open(r'E:\webots\MultiPati\controllers\tsattenGrasp\files\attLoss.txt', 'a') as file:
                        file.write(str(ci_shu) + " " + str(loss) + "\n")
                        file.close()

                with open(r'E:\webots\MultiPati\controllers\tsattenGrasp\files\attGoal.txt', 'a') as file:  # 如果当前轮次机器人成功抓取到目标梯级，则记录1到指定路径文本文档中，失败则记录0
                    file.write(str(goal) + " ")
                    file.close()

            if zhua == 1.0 or done == 1:  # 当机器人尝试抓取或者抓取中断时，重置训练环境，使用文本文档进行通信
                with open(file1_path + 'resetFlag.txt', 'r+') as file:
                    file.write('0')

                env.wait_reset(100)  # 等待一段时间用于通信
                env.reset()  # 仿真环境初始化
                ci_shu += 1
                break


if __name__ == '__main__':
    print("_________")
    with open(file1_path + 'resetFlag.txt', 'r+') as file:
        file.write('1')
    main()
