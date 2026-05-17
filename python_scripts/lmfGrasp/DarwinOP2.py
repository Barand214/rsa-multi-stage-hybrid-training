import sys
import math

import numpy as np
import torch
from PIL import Image

sys.path.append('D:/Dev/Tools/Webots/projects/robots/robotis/darwin-op/libraries/python37')
from controller import Robot, Motor, Motion, LED, Camera, Gyro, Accelerometer, PositionSensor, GPS
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager


robot = Robot()

class Darwin():
    def __init__(self):
        self.robot = robot  # 初始化Robot类以控制机器人
        # timeStep在webots界面的WorldInfo中设置,不能小于8，否则gaitManager不资瓷
        self.timeStep = int(self.robot.getBasicTimeStep())  # 获取当前每一个仿真步所仿真时间mTimeStep
        self.motionManager = RobotisOp2MotionManager(self.robot)  # 初始化机器人动作组控制器
        self.gaitManager = RobotisOp2GaitManager(self.robot, "config.ini")  # 初始化机器人步态控制器
        self.gaitManager.setBalanceEnable(True)

        #------------------------------------------启动传感器-----------------------------------
        self.motorNames = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                           'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL', 'PelvR',
                           'PelvL', 'LegUpperR', 'LegUpperL', 'LegLowerR', 'LegLowerL',
                           'AnkleR', 'AnkleL', 'FootR', 'FootL', 'Neck', 'Head', 'GraspL',
                           'GraspR')  # 机器人全身舵机名称，原有的20个＋新加的2个)
        self.PositionSensors = []  # 初始化关节角度传感器
        self.motors = []  # 初始化motor
        self.motorMinPositions = []  # 每个motor位置上界的编码值
        self.motorMaxPositions = []
        self.motorMaxTorque = []
        self.motorNum = len(self.motorNames)  # 有的论文会忽略Head motor的动作，这里未忽略。


        # self.keyboard = Keyboard()  # 键盘
        # self.keyboard.enable(self.timeStep)  # 采样周期
        # 约定设备变量的首字母大写，包括LED、Accelerometer、Gyro、Motors、PositionSensors
        self.HeadLED = self.robot.getDevice('HeadLed')  # 获取头部LED灯，高版本python需要用getDevice，低版本可以用getLED，下同
        self.EyeLED = self.robot.getDevice('EyeLed')  # 获取眼部LED灯
        self.HeadLED.set(0xff0000)  # 点亮头部LED灯并设置颜色
        self.EyeLED.set(0xa0a0ff)  # 点亮眼部LED灯并设置颜色

        self.Accelerometer = self.robot.getDevice('Accelerometer')  # 获取加速度传感器
        self.Accelerometer.enable(self.timeStep)  # 激活传感器，并以timeStep为周期更新数值
        self.Gyro = self.robot.getDevice('Gyro')  # 获取陀螺仪
        self.Gyro.enable(self.timeStep)  # 激活陀螺仪，并以timeStep为周期更新数值

        self.left_gps1 = self.robot.getDevice('left_gps1')  # 获取机器人GPS的设备对象
        self.right_gps1 = self.robot.getDevice('right_gps1')
        self.left_gps2 = self.robot.getDevice('left_gps2')
        self.right_gps2 = self.robot.getDevice('right_gps2')
        self.left_gps1.enable(self.timeStep)  # 激活机器人GPS，仿真时间步长为环境的最小仿真时间步长
        self.right_gps1.enable(self.timeStep)
        self.left_gps2.enable(self.timeStep)
        self.right_gps2.enable(self.timeStep)
        self.Camera = self.robot.getDevice('Camera')  # 获取机器人摄像头的设备对象
        self.Camera.enable(self.timeStep)  # 激活机器人摄像头，仿真时间步长为环境的最小仿真时间步长

        self.fup = 0
        self.fdown = 0  # 定义两个类变量，用于之后判断机器人是否摔倒

        self.deltaLimit = 0.2  # 如果motor的action为位置编码的增量，需要限定其幅度

        # 获取各传感器并激活，以mTimeStep为周期更新数值，获取各motor及其参数
        for i in range(0, self.motorNum):
            self.PositionSensors.append(self.robot.getDevice(self.motorNames[i] + 'S'))  # motor名称后加S即为传感器名称
            self.PositionSensors[i].enable(self.timeStep)
            self.motors.append(self.robot.getDevice(self.motorNames[i]))
            self.motorMinPositions.append(self.motors[i].getMinPosition())
            self.motorMaxPositions.append(self.motors[i].getMaxPosition())
            self.motorMaxTorque.append(self.motors[i].getMaxTorque())  # 读取各motor允许的最大扭矩，先将其discount再存储
            self.motors[i].setAvailableTorque(self.motorMaxTorque[i])  # 设置各motor运动时可以使用的最大扭矩



    # 按照webots中设定的timeStep，运行一个step
    def step(self):
        ret = self.robot.step(self.timeStep)
        if ret == -1:
            exit(0)

    # 等待，参数单位为s
    def wait(self, waitTime):
        startTime = self.robot.getTime()  # 通过设置一段时间，使得控制程序在此处循环等待，而仿真环境则可以一直运行
        s = waitTime / 1000
        while startTime + s >= self.robot.getTime():
            self.step()

    # 处理跌倒，目前暂时不管
    def checkFallen(self):  # 双足机器人在训练时一定经常跌倒，需要寻找一种高效的站起方法
        acc_tolerance = 60.0
        acc_step = 100  # 计数器上限
        acc = self.Accelerometer.getValues()  # 通过加速度传感器获取三轴的对应值
        if acc[1] < 512.0 - acc_tolerance:  # 面朝下倒地时y轴的值会变小
            self.fup += 1  # 计数器加1
        else:
            self.fup = 0  # 计数器清零
        if acc[1] > 512.0 + acc_tolerance:  # 背朝下倒地时y轴的值会变大
            self.fdown += 1  # 计数器加 1
        else:
            self.fdown = 0  # 计数器清零
        if self.fup > acc_step:  # 计数器超出上限，即倒地时间超过acc_step个仿真步长
            self.motionManager.playPage(10)  # 执行面朝下倒地起身动作
            self.motionManager.playPage(9)  # 恢复准备行走姿势
            self.fup = 0  # 计数器清零
        elif self.fdown > acc_step:
            self.motionManager.playPage(11)  # 执行背朝下倒地起身动作
            self.motionManager.playPage(9)  # 恢复准备行走姿势
            self.fdown = 0  # 计数器清零

    # 从加速度计读取加速度
    def getAcceleration(self):
        a = self.Accelerometer.getValues()
        return a

    # 根据读取的加速度计算机器人的倾斜角、滚转角
    def getAngle(self):
        a = self.getAcceleration()
        # Yaw通常使用其它传感器测量，但Darwin没有
        pitch = math.atan2(-a[0], math.sqrt(a[1] * a[1] + a[2] * a[2]))
        roll = math.atan2(a[1], a[2])
        return [pitch, roll]

    # 从陀螺仪读取角速度
    def getAngularVelocity(self):
        w = self.Gyro.getValues()
        return w

    # 获取20个motor的位置
    def getPosition(self):
        position = []
        for i in range(0, self.motorNum-2):  # 关节类型不同，测量得到以弧度为单位的角位置或以米为单位的线性位置
            position.append(self.PositionSensors[i].getValue())
        return position

    # 获取状态，目前还缺关节速度、偏航角
    def getState(self):  # 一部分状态，速度还需另外获取
        position = self.getPosition()  # dim=20
        angle = self.getAngle()  # dim=2
        w = self.getAngularVelocity()  # dim=3
        state = position
        state.extend(angle)
        state.extend(w)
        return state

    def robot_reset(self):   # 机器人舵机角度复位函数

        a = -np.random.random() * 0.4 + 0.7   # 动态参数，通过随机数在一定范围内，随机化机器人肩膀处的Shoulder舵机的初始角度
        b = -np.random.random() * 0.4 + 1.0   # 动态参数，通过随机数在一定范围内，随机化机器人手肘处的ArmLower舵机的初始角度
        self.step()
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
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()
        self.step()

    def print_gps(self):   # 获取当前GPS数值的函数
        gps_data1 = self.left_gps1.getValues()   # Webots2021.a版本使用.getValues()来获取设备对象的具体参数
        gps_data2 = self.right_gps1.getValues()
        gps_data3 = self.left_gps2.getValues()
        gps_data4 = self.right_gps2.getValues()
        return gps_data1, gps_data2, gps_data3, gps_data4   # 返回4个GPS数值

    def get_img(self, steps):   # 获取当前摄像头观察到图像的函数
        img = "img%s.png" % steps   # 为当前摄像头图像命名，将steps（当前训练轮次仿真步数）作为动态输入
        self.Camera.saveImage("imgs/"+img, 100)   # 使用软件内置函数保存图像，100为图像质量，范围为0-100，默认路径为控制器根目录
        # path = './%s' % img   # 获取保存图像的相对地址
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

    # def executePosition(self):








