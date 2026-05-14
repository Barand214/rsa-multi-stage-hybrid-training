import torch
import torch.nn as nn
import numpy as np
from controller import Robot, Motor, Motion, LED, Camera, Gyro, Accelerometer, PositionSensor, GPS
from PIL import Image
import time
import sys
import math

sys.path.append('D:/Webots/projects/robots/robotis/darwin-op/libraries/python37')
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager
from PPO.RobotRun1 import RobotRun
from replay_memory import ReplayMemory as replayMemory
from torch.distributions import Categorical
import copy

import torch.nn.functional as F

from Project_config import path_list

# GPU设置
# if  torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"

robot = Robot()

LR = 0.0001  # 学习率
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64  # 批处理的样本数量
GAMMA = 0.99  # 计算价值函数的折扣因子
TARGET_REPLACE_ITER = 100  # 目标网络的更新频率

PATH = "E:/Climb_Ladder/Climb_Ladder Project/controllers/Train_main/PPO"


class jie_duan1_Env():  # 子环境类
    def __init__(self):
        self.robot = robot
        self.state = None
        self.done = False
        self.isSuccess = False
        self.shooting = Shooting()

    def step(self, state, action, steps, zhua, gps1, gps2, gps3, gps4, name):  # 仿真平台更新函数
        return RobotRun(self.robot, state, action, steps, zhua, gps1, gps2, gps3, gps4, name).run()

    def reset(self):  # 重置状态函数
        self.robot_reset()  # 重置机器人状态
        with open(path_list['resetFlag'], 'r+') as file:  # 以记事本的形式传参，作为标识符
            file.write('0')
        with open(path_list['resetFlag1'], 'r+') as file:
            file.write('0')
        self.done = False
        return self.state

    def robot_reset(self):  # 重置机器人的舵机角度参数
        return self.shooting.robot_reset()

    def print_gps(self):  # 获取机器人GPS当前的位置参数
        return self.shooting.print_gps()

    def get_img(self, steps):  # 获取机器人摄像头观察到的图像信息
        return self.shooting.get_img(steps, 1)

    def get_robot_state(self):  # 获取机器人的当前的舵机角度参数
        return self.shooting.get_robot_state()

    def wait_reset(self, s):  # 暂时停止控制代码向下运行，为仿真环境中机器人舵机运动留出一定时间
        return self.shooting.wait(s)


class Shooting():  # 总环境类
    def __init__(self):
        self.timestep = int(robot.getBasicTimeStep())  # 初始化环境的最小仿真时间步长，具体数值为仿真环境中设置的大小
        self.gaitManager = RobotisOp2GaitManager(robot, 'config.ini')  # 初始化Robotis-OP2机器人的步态控制器，确定参数配置文件
        self.motionManager = RobotisOp2MotionManager(robot)
        self.gaitManager.setBalanceEnable(True)

        # --------------------------------启动传感器----------------------------------
        self.motors = []  # 机器人舵机名称列表初始化
        self.motors_sensors = []  # 机器人舵机传感器列表初始化
        self.motorName = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                          'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL', 'PelvR',
                          'PelvL', 'LegUpperR', 'LegUpperL', 'LegLowerR', 'LegLowerL',
                          'AnkleR', 'AnkleL', 'FootR', 'FootL', 'Neck', 'Head', 'GraspL',
                          'GraspR')  # 机器人全身舵机名称，原有的20个＋新加的2个
        self.eyeLed = robot.getDevice('EyeLed')  # 获取机器人眼部LED灯的设备对象
        self.headLed = robot.getDevice('HeadLed')  # 获取机器人头部LE灯的设备对象
        self.camera = robot.getDevice('Camera')  # 获取机器人摄像头的设备对象
        self.accelerometer = robot.getDevice('Accelerometer')  # 获取机器人加速度测量仪的设备对象
        self.gyro = robot.getDevice('Gyro')  # 获取机器人陀螺仪的设备对象
        self.left_gps1 = robot.getDevice('left_gps1')  # 获取机器人GPS的设备对象
        self.right_gps1 = robot.getDevice('right_gps1')
        self.left_gps2 = robot.getDevice('left_gps2')
        self.right_gps2 = robot.getDevice('right_gps2')
        self.left_gps1.enable(self.timestep)  # 激活机器人GPS，仿真时间步长为环境的最小仿真时间步长
        self.right_gps1.enable(self.timestep)
        self.left_gps2.enable(self.timestep)
        self.right_gps2.enable(self.timestep)
        self.camera.enable(self.timestep)  # 激活机器人摄像头，仿真时间步长为环境的最小仿真时间步长
        self.accelerometer.enable(self.timestep)  # 激活机器人加速度测量仪，仿真时间步长为环境的最小仿真时间步长
        self.gyro.enable(self.timestep)  # 激活机器人陀螺仪，仿真时间步长为环境的最小仿真时间步长
        for i in range(len(self.motorName)):  # 依次处理机器人的全身舵机
            self.motors.append(robot.getDevice(self.motorName[i]))  # 根据舵机名称获取舵机的设备对象，将其放入motors列表中
            sensorName = self.motorName[i]
            sensorName = sensorName + 'S'  # Webots中舵机传感器的名称是舵机名称+S
            self.position = robot.getDevice(sensorName)  # 根据舵机传感器名称获取舵机传感器的设备对象
            self.position.enable(self.timestep)  # 激活舵机传感器
            self.motors_sensors.append(self.position)  # 将激活后的舵机传感器放入motor_sensors列表

        # ---------------------------------启动结束-----------------------------------

    def myStep(self):  # 单步仿真函数
        robot.step(self.timestep)  # 每次调用该函数，仿真环境就进行1个最小时间步长的运行

    def wait(self, ms):  # 时间段仿真函数
        startTime = robot.getTime()  # 通过设置一段时间，使得控制程序在此处循环等待，而仿真环境则可以一直运行
        s = ms / 1000
        while s + startTime >= robot.getTime():
            self.myStep()

    def robot_reset(self):  # 机器人舵机角度复位函数
        a = -np.random.random() * 0.4 + 0.7  # 动态参数，通过随机数在一定范围内，随机化机器人肩膀处的Shoulder舵机的初始角度
        b = -np.random.random() * 0.4 + 1.0  # 动态参数，通过随机数在一定范围内，随机化机器人手肘处的ArmLower舵机的初始角度
        self.myStep()
        # 优先初始化手部夹爪的舵机角度参数
        self.motors[20].setPosition(1)  # left   # 通过直接调用软件自带的setPosition函数，可以直接控制舵机的设备对象，使其以默认速度转动到指定角度
        self.motors[21].setPosition(1)  # right)
        self.wait(200)  # 使代码等待一段时间，直到仿真环境机器人手部夹爪舵机结束运动
        self.gaitManager.stop()  # 暂停机器人步态控制器
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
        self.motors[19].setVelocity(1)  # head
        self.motors[7].setVelocity(1)  # PelvYL
        self.motors[9].setVelocity(1)  # PelvL
        self.motors[11].setVelocity(1)  # LegUpperL
        self.motors[13].setVelocity(1)  # LegLowerL
        self.motors[15].setVelocity(1)  # AnkleL
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
        print("环境重置")

    def print_gps(self):  # 获取当前GPS数值的函数
        gps_data1 = self.left_gps1.getValues()  # Webots2021.a版本使用.getValues()来获取设备对象的具体参数
        gps_data2 = self.right_gps1.getValues()
        gps_data3 = self.left_gps2.getValues()
        gps_data4 = self.right_gps2.getValues()
        return gps_data1, gps_data2, gps_data3, gps_data4  # 返回4个GPS数值

    def get_img(self, steps, imgs):  # 获取当前摄像头观察到图像的函数
        img = "img%s.png" % steps  # 为当前摄像头图像命名，将steps（当前训练轮次仿真步数）作为动态输入
        self.camera.saveImage(img, 100)  # 使用软件内置函数保存图像，100为图像质量，范围为0-100，默认路径为控制器根目录
        path = './%s' % img  # 获取保存图像的相对地址
        img = Image.open(path)  # 根据路径名称打开所保存的图像
        img = img.resize((128, 128))  # 调整保存图像的大小，仿真环境中摄像头获取到的图像大小为120X160，也可以设置为240X320等尺寸
        img = img.convert('L')  # 对摄像头图像进行灰度化处理
        img = np.array(img)  # 转化摄像头图像为数组形式
        img = img / 225.0  # 对摄像头图像进行归一化操作
        img_tensor = torch.tensor(img)  # 转化摄像头图像为张量形式
        img_tensor = torch.unsqueeze(img_tensor, 0)  # 增加张量维度
        img_tensor = img_tensor.float()  # 将张量转换为浮点数类型的张量。
        return img, img_tensor

    def get_robot_state(self):  # 获取当前机器人舵机角度参数
        self.robot_state = []  # 建立用于储存数据的列表
        for i in range(len(self.motorName) - 2):  # 通过循环的形式收集原有的20个舵机的角度参数
            position = self.motors_sensors[i].getValue()
            self.robot_state.append(position)
        return self.robot_state

    """
       接下来为固定动作组的代码，每一个函数都代表着一个固定动作组的实现
       每个固定动作组在结构上均类似，具体参数则通过手动调节获得
    """

    def tai_leg_L1(self):
        timer = 0
        L_motor1 = robot.getDevice('LegLowerL')  # 获取设备对象
        L_motor1.setPosition(-0.7)  # 设置舵机目标角度
        L_motor1.setVelocity(1)  # 设置舵机运动速度
        L_motor2 = robot.getDevice('AnkleL')
        L_motor2.setPosition(-0.5)
        L_motor2.setVelocity(1)
        while robot.step(32) != -1:  # 设置循环等待时间
            timer += 32
            if timer >= 2000:
                break

    def tai_leg_L2(self):
        timer = 0
        L_motor1 = robot.getDevice('LegUpperL')
        L_motor1.setPosition(1.65)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('LegLowerL')
        L_motor2.setPosition(-2.2)
        L_motor2.setVelocity(2)
        L_motor3 = robot.getDevice('AnkleL')
        L_motor3.setPosition(-0.85)
        L_motor3.setVelocity(2)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 2000:
                break

    def tai_leg_L3(self):
        timer = 0
        L_motor1 = robot.getDevice('LegLowerL')
        L_motor1.setPosition(-1.8)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('AnkleL')
        L_motor2.setPosition(-0.45)
        L_motor2.setVelocity(1)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 1000:
                break

    def tai_leg_L4(self):
        timer = 0
        L_motor1 = robot.getDevice('LegUpperL')
        L_motor1.setPosition(1.4)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('LegLowerL')
        L_motor2.setPosition(-1.55)
        L_motor2.setVelocity(1)
        L_motor3 = robot.getDevice('AnkleL')
        L_motor3.setPosition(-0.45)
        L_motor3.setVelocity(1)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 1000:
                break

    def n_tai_leg_L1(self):
        timer = 0
        L_motor1 = robot.getDevice('LegLowerL')
        L_motor1.setPosition(-1.45)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('AnkleL')
        L_motor2.setPosition(-0.45)
        L_motor2.setVelocity(1)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 2000:
                break

    def n_tai_leg_L2(self):
        timer = 0
        L_motor1 = robot.getDevice('LegUpperL')
        L_motor1.setPosition(1.65)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('LegLowerL')
        L_motor2.setPosition(-2.2)
        L_motor2.setVelocity(2)
        L_motor3 = robot.getDevice('AnkleL')
        L_motor3.setPosition(-0.85)
        L_motor3.setVelocity(2)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 2000:
                break

    def n_tai_leg_L3(self):
        timer = 0
        L_motor1 = robot.getDevice('LegLowerL')
        L_motor1.setPosition(-1.8)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('AnkleL')
        L_motor2.setPosition(-0.8)
        L_motor2.setVelocity(1)
        L_motor3 = robot.getDevice('AnkleR')
        L_motor3.setPosition(-0.8)
        L_motor3.setVelocity(1)
        L_motor4 = robot.getDevice('ArmLowerL')
        L_motor4.setPosition(0.2)
        L_motor4.setVelocity(1)
        L_motor5 = robot.getDevice('ShoulderL')
        L_motor5.setPosition(-0.4)
        L_motor5.setVelocity(1)
        L_motor6 = robot.getDevice('ArmLowerR')
        L_motor6.setPosition(-0.2)
        L_motor6.setVelocity(1)
        L_motor7 = robot.getDevice('ShoulderR')
        L_motor7.setPosition(0.4)
        L_motor7.setVelocity(1)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 2000:
                break

    def n_tai_leg_L4(self):
        timer = 0
        L_motor2 = robot.getDevice('LegLowerL')
        L_motor2.setPosition(-1.6)
        L_motor2.setVelocity(1)
        L_motor3 = robot.getDevice('AnkleL')
        L_motor3.setPosition(-0.45)
        L_motor3.setVelocity(1)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 1000:
                break

    def song_L(self):
        timer = 0
        L_motor1 = robot.getDevice('GraspL')
        L_motor1.setPosition(1)
        L_motor1.setVelocity(2)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 1000:
                break

    def song_R(self):
        timer = 0
        L_motor1 = robot.getDevice('GraspR')
        L_motor1.setPosition(1)
        L_motor1.setVelocity(2)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 1000:
                break

    def tai_leg_R1(self):
        timer = 0
        L_motor1 = robot.getDevice('LegLowerR')
        L_motor1.setPosition(0.7)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('AnkleR')
        L_motor2.setPosition(0.35)
        L_motor2.setVelocity(1)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 2000:
                break

    def tai_leg_R2(self):
        timer = 0
        L_motor1 = robot.getDevice('LegUpperR')
        L_motor1.setPosition(-1.65)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('LegLowerR')
        L_motor2.setPosition(2.2)
        L_motor2.setVelocity(2)
        L_motor3 = robot.getDevice('AnkleR')
        L_motor3.setPosition(0.85)
        L_motor3.setVelocity(2)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 2000:
                break

    def tai_leg_R3(self):
        timer = 0
        L_motor1 = robot.getDevice('LegLowerR')
        L_motor1.setPosition(1.8)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('AnkleR')
        L_motor2.setPosition(0.45)
        L_motor2.setVelocity(1)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 1000:
                break

    def tai_leg_R4(self):
        timer = 0
        L_motor1 = robot.getDevice('LegUpperR')
        L_motor1.setPosition(-1.4)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('LegLowerR')
        L_motor2.setPosition(1.55)
        L_motor2.setVelocity(1)
        L_motor3 = robot.getDevice('AnkleR')
        L_motor3.setPosition(0.45)
        L_motor3.setVelocity(1)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 1000:
                break

    def n_tai_leg_R1(self):
        timer = 0
        L_motor1 = robot.getDevice('LegLowerR')
        L_motor1.setPosition(1.45)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('AnkleR')
        L_motor2.setPosition(0.45)
        L_motor2.setVelocity(1)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 2000:
                break

    def n_tai_leg_R2(self):
        timer = 0
        L_motor1 = robot.getDevice('LegUpperR')
        L_motor1.setPosition(-1.65)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('LegLowerR')
        L_motor2.setPosition(2.2)
        L_motor2.setVelocity(2)
        L_motor3 = robot.getDevice('AnkleR')
        L_motor3.setPosition(0.85)
        L_motor3.setVelocity(2)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 2000:
                break

    def n_tai_leg_R3(self):
        timer = 0
        L_motor1 = robot.getDevice('LegLowerR')
        L_motor1.setPosition(1.8)
        L_motor1.setVelocity(1)
        L_motor3 = robot.getDevice('AnkleR')
        L_motor3.setPosition(0.8)
        L_motor3.setVelocity(1)
        L_motor4 = robot.getDevice('ArmLowerL')
        L_motor4.setPosition(-0.2)
        L_motor4.setVelocity(1)
        L_motor5 = robot.getDevice('ShoulderL')
        L_motor5.setPosition(-0.4)
        L_motor5.setVelocity(1)
        L_motor6 = robot.getDevice('ArmLowerR')
        L_motor6.setPosition(-0.2)
        L_motor6.setVelocity(1)
        L_motor7 = robot.getDevice('ShoulderR')
        L_motor7.setPosition(0.4)
        L_motor7.setVelocity(1)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 2000:
                break

    def n_tai_leg_R4(self):
        timer = 0
        L_motor2 = robot.getDevice('LegLowerR')
        L_motor2.setPosition(1.6)
        L_motor2.setVelocity(1)
        L_motor3 = robot.getDevice('AnkleR')
        L_motor3.setPosition(0.45)
        L_motor3.setVelocity(1)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 1000:
                break

    def tai_arm_L1(self):
        timer = 0
        L_motor1 = robot.getDevice('ArmLowerL')
        L_motor1.setPosition(-0.8)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('ArmUpperL')
        L_motor2.setPosition(0.65)
        L_motor2.setVelocity(1)
        L_motor3 = robot.getDevice('ShoulderL')
        L_motor3.setPosition(0.5)
        L_motor3.setVelocity(1)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 1000:
                break

    def tai_arm_L2(self):
        timer = 0
        L_motor1 = robot.getDevice('ArmLowerL')
        L_motor1.setPosition(-1)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('Head')
        L_motor2.setPosition(0.4)
        L_motor2.setVelocity(1)
        L_motor3 = robot.getDevice('ShoulderL')
        L_motor3.setPosition(0.2)
        L_motor3.setVelocity(1)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 1000:
                break

    def tai_arm_R1(self):
        timer = 0
        L_motor1 = robot.getDevice('ArmLowerR')
        L_motor1.setPosition(0.8)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('ArmUpperR')
        L_motor2.setPosition(-0.65)
        L_motor2.setVelocity(1)
        L_motor3 = robot.getDevice('ShoulderR')
        L_motor3.setPosition(-0.5)
        L_motor3.setVelocity(1)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 1000:
                break

    def tai_arm_R2(self):
        timer = 0
        L_motor1 = robot.getDevice('ArmLowerR')
        L_motor1.setPosition(1)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('Head')
        L_motor2.setPosition(0.4)
        L_motor2.setVelocity(1)
        L_motor3 = robot.getDevice('ShoulderR')
        L_motor3.setPosition(-0.2)
        L_motor3.setVelocity(1)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 1000:
                break

    def tiao_zheng(self):
        timer = 0
        L_motor1 = robot.getDevice('ArmLowerL')
        L_motor1.setPosition(-0.7)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('ArmLowerR')
        L_motor2.setPosition(0.7)
        L_motor2.setVelocity(1)
        L_motor3 = robot.getDevice('ShoulderL')
        L_motor3.setPosition(0)
        L_motor3.setVelocity(1)
        L_motor4 = robot.getDevice('ShoulderR')
        L_motor4.setPosition(0)
        L_motor4.setVelocity(1)
        L_motor5 = robot.getDevice('AnkleL')
        L_motor5.setPosition(-0.55)
        L_motor5.setVelocity(1)
        L_motor6 = robot.getDevice('AnkleR')
        L_motor6.setPosition(0.55)
        L_motor6.setVelocity(1)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 1000:
                break

    def sheng_gao1(self):
        timer = 0
        L_motor1 = robot.getDevice('ShoulderL')
        L_motor1.setPosition(-1.3)
        L_motor1.setVelocity(2)
        L_motor2 = robot.getDevice('ShoulderR')
        L_motor2.setPosition(1.3)
        L_motor2.setVelocity(2)
        L_motor3 = robot.getDevice('ArmLowerL')
        L_motor3.setPosition(1)
        L_motor3.setVelocity(2)
        L_motor4 = robot.getDevice('ArmLowerR')
        L_motor4.setPosition(-1)
        L_motor4.setVelocity(2)
        L_motor5 = robot.getDevice('LegLowerL')
        L_motor5.setPosition(-1.2)
        L_motor5.setVelocity(2)
        L_motor6 = robot.getDevice('LegLowerR')
        L_motor6.setPosition(1.2)
        L_motor6.setVelocity(2)
        L_motor7 = robot.getDevice('LegUpperL')
        L_motor7.setPosition(1.45)
        L_motor7.setVelocity(2)
        L_motor8 = robot.getDevice('LegUpperR')
        L_motor8.setPosition(-1.45)
        L_motor8.setVelocity(2)
        L_motor9 = robot.getDevice('AnkleL')
        L_motor9.setPosition(-0.25)
        L_motor9.setVelocity(2)
        L_motor10 = robot.getDevice('AnkleR')
        L_motor10.setPosition(0.25)
        L_motor10.setVelocity(2)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 2000:
                break

    def la_jin(self):
        timer = 0
        L_motor1 = robot.getDevice('ArmLowerL')
        L_motor1.setPosition(0.5)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('ArmLowerR')
        L_motor2.setPosition(-0.5)
        L_motor2.setVelocity(1)
        L_motor3 = robot.getDevice('ShoulderL')
        L_motor3.setPosition(-0.7)
        L_motor3.setVelocity(1)
        L_motor4 = robot.getDevice('ShoulderR')
        L_motor4.setPosition(0.7)
        L_motor4.setVelocity(2)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 1000:
                break

    def tiao_zheng1(self):
        timer = 0
        L_motor1 = robot.getDevice('ArmLowerL')
        L_motor1.setPosition(0)
        L_motor1.setVelocity(1)
        L_motor2 = robot.getDevice('ArmLowerR')
        L_motor2.setPosition(0)
        L_motor2.setVelocity(1)
        L_motor3 = robot.getDevice('ShoulderL')
        L_motor3.setPosition(0)
        L_motor3.setVelocity(1)
        L_motor4 = robot.getDevice('ShoulderR')
        L_motor4.setPosition(0)
        L_motor4.setVelocity(1)
        L_motor5 = robot.getDevice('AnkleL')
        L_motor5.setPosition(-0.55)
        L_motor5.setVelocity(1)
        L_motor6 = robot.getDevice('AnkleR')
        L_motor6.setPosition(0.55)
        L_motor6.setVelocity(1)
        L_motor7 = robot.getDevice('LegLowerL')
        L_motor7.setPosition(-1.57)
        L_motor7.setVelocity(1)
        L_motor8 = robot.getDevice('LegLowerR')
        L_motor8.setPosition(1.57)
        L_motor8.setVelocity(1)
        L_motor9 = robot.getDevice('LegUpperL')
        L_motor9.setPosition(1.4)
        L_motor9.setVelocity(1)
        L_motor10 = robot.getDevice('LegUpperR')
        L_motor10.setPosition(-1.4)
        L_motor10.setVelocity(1)
        while robot.step(32) != -1:
            timer += 32
            if timer >= 2000:
                break


class Net_act(nn.Module):  # 网络参数类，使用pytorch框架，并使用CUDA加速
    def __init__(self, act_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        self.relu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        self.fc0 = nn.Linear(in_features=6272, out_features=6000)
        self.fc1 = nn.Linear(in_features=6000, out_features=100)
        self.fc2 = nn.Linear(in_features=20, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=100)
        self.fc4 = nn.Linear(in_features=200, out_features=200)
        self.fc5 = nn.Linear(in_features=200, out_features=act_dim)
        self.softmax = nn.Softmax(dim=-1)  # Softmax 激活函数

    def forward(self, x, state):  # 神经网络前向传播
        x = torch.tensor(x).to('cuda')  # 参数转化为张量并启用CUDA，如不使用，将'cuda'删除即可
        x = torch.unsqueeze(x, dim=0)  # 张量扩维
        x = x.float()  # 张量浮点化
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = torch.flatten(x)
        # state = torch.tensor(state).to('cuda')
        # state = state.float()
        x = torch.flatten(x)
        x = x.float()
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        state = self.relu(self.fc2(state))
        state = self.relu(self.fc3(state))

        min_val1 = torch.min(x)
        max_val1 = torch.max(x)
        normalized_data1 = torch.div(torch.sub(x, min_val1), torch.sub(max_val1, min_val1))  # 最大最小标准化
        min_val2 = torch.min(state)
        max_val2 = torch.max(state)
        normalized_data2 = torch.div(torch.sub(state, min_val2), torch.sub(max_val2, min_val2))

        state_x = torch.cat((normalized_data1, normalized_data2), dim=-1)  # 状态和动作张量相连接
        state_x = self.relu(self.fc4(state_x))
        state_x = self.fc5(state_x)
        state_x = self.softmax(state_x)
        return state_x

    # def forward(self, x, state):
    #     state = F.relu(self.fc2(state))
    #     state = F.relu(self.fc3(state))
    #     probs = F.softmax(self.fc5(state), dim=-1)
    #     return probs


class Net_value(nn.Module):  # 网络参数类，使用pytorch框架，并使用CUDA加速
    def __init__(self, act_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        self.relu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        self.fc0 = nn.Linear(in_features=6272, out_features=6000)
        self.fc1 = nn.Linear(in_features=6000, out_features=100)
        self.fc2 = nn.Linear(in_features=20, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=100)
        self.fc4 = nn.Linear(in_features=200, out_features=200)
        self.fc5 = nn.Linear(in_features=200, out_features=act_dim)

    def forward(self, x, state):  # 神经网络前向传播
        x = torch.tensor(x).to('cuda')  # 参数转化为张量并启用CUDA，如不使用，将'cuda'删除即可
        x = torch.unsqueeze(x, dim=0)  # 张量扩维
        x = x.float()  # 张量浮点化
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = torch.flatten(x)
        # state = torch.tensor(state).to('cuda')
        # state = state.float()
        x = torch.flatten(x)
        x = x.float()
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        state = self.relu(self.fc2(state))
        state = self.relu(self.fc3(state))

        min_val1 = torch.min(x)
        max_val1 = torch.max(x)
        normalized_data1 = torch.div(torch.sub(x, min_val1), torch.sub(max_val1, min_val1))  # 最大最小标准化
        min_val2 = torch.min(state)
        max_val2 = torch.max(state)
        normalized_data2 = torch.div(torch.sub(state, min_val2), torch.sub(max_val2, min_val2))

        state_x = torch.cat((normalized_data1, normalized_data2), dim=-1)  # 状态和动作张量相连接
        state_x = self.relu(self.fc4(state_x))
        state_x = self.fc5(state_x)
        return state_x

    # def forward(self, x, state):
    #     x = F.relu(self.fc2(state))
    #     x = F.relu(self.fc3(x))
    #     value = self.fc5(x)
    #     return value


class PPO(object):  # PPO算法类
    def __init__(self):
        self.lr = 0.0005  # 网络学习率
        self.ok_num = 1000  # 从第几次训练开始选择概率较大的动作而不是根据概率随机选择

        self.gamma = 0.99  # 折扣因子
        self.k_epochs = 4  # 更新策略网络的次数
        self.eps_clip = 0.2  # epsilon-clip
        self.entropy_coef = 0.01  # entropy的系数
        # 创建评估网络和目标网络
        self.target_net = Net_act(2).to('cuda')  # 初始化两个结构相同的神经网络，一个作为评估网络，一个作为目标网络
        self.eval_net = Net_value(1).to('cuda')
        self.actor_optimizer = torch.optim.Adam(self.target_net.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

        self.learn_step_counter = 0  # 学习步数记录
        self.memory_counter = 0  # 记忆量计数
        self.memory = np.zeros((MEMORY_CAPACITY, 6))  # 存储空间初始化，每一组的数据为(o_t,s_t,a_t,r_t,o_{t+1},s_{t+1})
        self.optimazer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # 初始化优化器
        self.loss_func = nn.MSELoss()  # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.loss_func = self.loss_func.to()

        self.old_log_probs = []

    def choose_action(self, num, x, y):  # 定义动作选择函数 (x为图像，y为状态)

        y = torch.tensor(y).to('cuda')
        y = y.float()

        actions_value = self.target_net.forward(x, y)  # 通过对评估网络输入图像x和状态y，前向传播获得动作值

        if num < self.ok_num:
            dist = Categorical(actions_value)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            # 使用 softmax 获取动作的概率分布
            actions_prob = torch.softmax(actions_value, dim=-1)

            # 获取概率最大的动作
            action = torch.argmax(actions_prob, dim=-1)

            # 获取该动作的 log 概率
            log_prob = torch.log(actions_prob[action])

        return action, log_prob.detach()

    def learn(self, rpm, sum, n):  # 神经网络参数训练函数
        if (len(rpm) < sum):
            return

        print("开始学习")
        obs, old_states, old_actions, old_log_probs, old_rewards, old_dones = rpm.sample(
            n)  # 从样本经验池随机选取64组样本用于训练
        # obs = torch.tensor(np.array(obs), device="cuda", dtype=torch.float32)
        old_states = torch.tensor(np.array(old_states), device="cuda", dtype=torch.float32)
        old_actions = torch.tensor(np.array(old_actions), device="cuda", dtype=torch.float32)
        # obs = torch.tensor(np.array(obs), device="cuda", dtype=torch.float32)
        # old_log_probs = self.old_log_probs[-n:]
        old_log_probs = torch.tensor(old_log_probs, device="cuda", dtype=torch.float32)

        # print(old_states)

        # old_log_probs = torch.stack(old_log_probs)

        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(old_rewards), reversed(old_dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        # Normalizing the rewards:
        returns = torch.tensor(returns, device="cuda", dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # 1e-5 to avoid division by zero

        for _ in range(self.k_epochs):
            # compute advantage
            values = []  # detach to avoid backprop through the critic
            for i, j in zip(obs, old_states):
                values.append(self.eval_net(i, j))
            values = torch.stack(values)

            advantage = returns - values.detach()
            # get action probabilities
            probs = []
            for i, j in zip(obs, old_states):
                probs.append(self.target_net(i, j))
            probs = torch.stack(probs)
            # print(probs)

            dist = Categorical(probs)
            # get new action probabilities
            new_probs = dist.log_prob(old_actions)
            # compute ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(new_probs - old_log_probs.detach())  # old_log_probs must be detached
            # compute surrogate loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            # compute actor loss
            actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()
            # actor_loss = -torch.min(surr1, surr2) + self.entropy_coef * dist.entropy().mean()
            # print(actor_loss)
            # compute critic loss
            critic_loss = (returns - values).pow(2).mean()

            # print(actor_loss, critic_loss)
            # critic_loss =-torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()+0.5*(returns - values).pow(2).mean()
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()  # 没有 retain_graph=True
            critic_loss.backward()  # 没有 retain_graph=True
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        rpm.clear()
        return 1


def success(Flag, pos, number):
    num = 0
    num_1 = 0
    if pos - number - 1 < 0:
        temp = Flag[0:pos + number]
        num = pos + number
    else:
        temp = Flag[pos - number - 1:pos + number]
        num = 2 * number + 1
    for i in temp:
        if i == "1":
            num_1 += 1
    return num_1 / num * 100


def PPO_episoid():  # 主函数
    Flag = ""
    temp_1 = []
    temp_2 = []
    save = 100  # 多少次保存一次模型
    eval_eps = 10  # 多少次训练后评估模型并更新
    eval_step = 5  # 评估的轮数
    sum_train = 5000  # 总训练伦数
    pos_1 = []  # 这两个空列表是用来记录和保存模型
    pos_2 = []
    ans_success = []  # 保存成功率
    min_success = 90  # 设置阈值，准确率高于这个值的模型才进行保存
    max_success = 0  # 记录最大的成功率
    ans_max_success_pos = 0  # 几率最大成功了的模型位置
    ppo = PPO()
    best_ppo = None  # 记录最佳模型
    ci_shu = 0  # 机器人执行动作的次数
    best_ep_reward = 0  # 记录最大回合奖励
    rpm = replayMemory(1000)
    sum_rpm = 200  # 一共收集多少数据
    use_rpm = 64  # 采用多少收集的数据进行训练，理论上是越大越好，但是这个会影响训练速度
    gps_goal = [0.2, 0.165]  # 设置目标基准坐标点（抓取梯子的坐标）
    for train_num in range(sum_train):  # 设置训练总轮次
        env = jie_duan1_Env()  # 环境初始化
        print("训练：第 " + str(train_num + 1) + " / " + str(sum_train) + " 轮")
        ep_reward = 0  # 记录一回合内的奖励
        step = 0
        env.reset()
        env.wait_reset(500)
        obs, obs_tensor = env.get_img(step)
        robot_state = env.get_robot_state()  # 重置环境，返回初始状态
        while True:
            step += 1
            obs, obs_tensor = env.get_img(step)
            action, log_probs = ppo.choose_action(train_num ,obs, robot_state)  # 选择动作
            gps1, gps2, gps3, gps4 = env.print_gps()  # 获取当前GPS坐标参数
            if step >= 29:  # 设置每轮训练的最大运行步数，超过运行步数强行终止机器人运动，夹爪立即关闭 原本为29
                zhua = 1.0
            else:
                zhua = 0.0
            name = "img" + str(step) + ".png"
            next_state, reward, done, good, goal, count = env.step(robot_state, action, step, zhua, gps1, gps2, gps3,
                                                                   gps4, name)  # 环境更新
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

            if good == 1:  # 如果此次机器人运行正常，将合格的数据储存到样本经验回放池中
                rpm.append((obs, robot_state, action, log_probs, reward, done))  # 保存transition

            robot_state = next_state  # 更新下一个状态
            if done == 1:
                loss = ppo.learn(rpm, sum_rpm, use_rpm)  # 模型学习
                ep_reward += reward  # 累加奖励

                if goal == 1 and ci_shu % save != 0:  # 当机器人夹爪成功抓取到目标梯级时，将当前网络参数储存到指定位置，用于测试使用
                    save_path = 'checkpoint/dqn_model_%s.ckpt' % ci_shu
                    ppo1 = best_ppo
                    temp_2.append(ppo1)
                    pos_2.append(ci_shu % save)

                if ci_shu % save == 0 and ci_shu != 0:  # 每100次训练，更新目标网络参数，并将当前网络参数储存到指定位置，用于测试使用
                    ppo1 = best_ppo
                    pos_2.append(save)
                    temp_2.append(ppo1)
                    if ci_shu != save:
                        with open(PATH + 'success.txt', 'a') as file:
                            for i in pos_1:
                                file.write(
                                    f"第{ci_shu - 2 * save + i}次训练模型的准确率大致为:{success(Flag, ci_shu - 2 * save + i, 10)}%\n")
                                ans_success.append(success(Flag, ci_shu - 2 * save + i, 10))
                                if success(Flag, ci_shu - 2 * save + i, 10) > max_success:
                                    max_success = success(Flag, ci_shu - 2 * save + i, 10)
                                    ans_max_success_pos = ci_shu - 2 * save + i
                            file.close()
                        for i in range(len(pos_1)):
                            if ans_success[i] > min_success:
                                save_path = 'checkpoint/dqn_model_%s.ckpt' % (ci_shu - 2 * save + pos_1[i])
                                torch.save(temp_1[i].eval_net, save_path)
                    temp_1 = temp_2
                    pos_1 = pos_2
                    pos_2.clear()
                    temp_2.clear()
                    ans_success.clear()

                ci_shu += 1

                with open(PATH + 'return.txt', 'a') as file:
                    return_all_str = str(ep_reward)
                    file.write(return_all_str)
                    file.write(",")
                    file.close()
                with open(PATH + 'goal.txt',
                          'a') as file:  # 如果当前轮次机器人成功抓取到目标梯级，则记录1到指定路径文本文档中，失败则记录0
                    goal_str = str(goal)
                    Flag += goal_str
                    file.write(goal_str)
                    file.write(",")
                    file.close()

            if done == 1 or zhua == 1:
                with open(path_list['resetFlag'], 'r+') as file:
                    file.write('0')
                env.wait_reset(100)  # 等待一段时间用于通信
                env.reset()  # 仿真环境初始化
                step = 0
                # ci_shu = ci_shu + 1
                obs, obs_tensor = env.get_img(step)
                robot_state = env.get_robot_state()
                break

        if train_num % eval_eps == 0:
            sum_eval_reward = 0
            print("评估开始")
            # print(train_num)
            for j in range(eval_step):
                print("评估：第 " + str(j + 1) + " / " + str(eval_step) + " 轮")
                eval_ep_reward = 0
                env.reset()
                env.wait_reset(500)
                obs, obs_tensor = env.get_img(step)
                robot_state = env.get_robot_state()
                step = 0
                while True:
                    step += 1
                    action, log_prob = ppo.choose_action(train_num ,obs, robot_state)  # 选择动作
                    action = action.detach().cpu().numpy().item()

                    gps1, gps2, gps3, gps4 = env.print_gps()  # 获取当前GPS坐标参数
                    if step >= 29:  # 设置每轮训练的最大运行步数，超过运行步数强行终止机器人运动，夹爪立即关闭 原本为29
                        zhua = 1.0
                    else:
                        zhua = 0.0
                    name = "img" + str(step) + ".png"
                    next_state, reward, done, good, goal, count = env.step(robot_state, action, step, zhua,
                                                                           gps1, gps2, gps3, gps4, name)  # 环境更新
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

                    robot_state = next_state  # 更新下一个状态
                    eval_ep_reward += reward  # 累加奖励
                    if done == 1 or zhua == 1:
                        break
                sum_eval_reward += eval_ep_reward
            mean_eval_reward = sum_eval_reward / eval_step
            if mean_eval_reward >= best_ep_reward:
                best_ep_reward = mean_eval_reward
                best_ppo = copy.deepcopy(ppo)
                print(
                    f"回合：{train_num + 1}，评估奖励：{mean_eval_reward:.2f}，最佳评估奖励：{best_ep_reward:.2f}，更新模型！")
            else:
                print(
                    f"回合：{train_num + 1}，评估奖励：{mean_eval_reward:.2f}，最佳评估奖励：{best_ep_reward:.2f}")

    for i in range(len(pos_1)):
        save_path = 'checkpoint/dqn_model_%s.ckpt' % (ci_shu - 2 * save + pos_1[i])
        torch.save(temp_1[i].eval_net, save_path)
    with open(PATH + 'success.txt', 'a') as file:
        for i in pos_1:
            file.write(
                f"第{ci_shu - 2 * save + i}次训练模型的准确率大致为:{success(Flag, ci_shu - 2 * save + i, 10)}%\n")
        file.close()

    with open(PATH + 'success.txt', 'a') as file:
        file.write(f"最大模型成功率为{max_success}  最优模型位置为{ans_max_success_pos}")
        file.close()


if __name__ == '__main__':
    print("_________")
    with open(path_list['resetFlag'], 'r+') as file:
        file.write('1')
    main()
