"""RobotRun controller."""
import math

#import gym
import time
import numpy as np
import os
import cv2

import argparse
import platform
import sys
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class RobotRun:
    # 控制机器人按照action行动的类
    # action:
    def __init__(self, robot, state, action, step, zhua, gps1, gps2, gps3, gps4, name):
        self.name = name
        self.robot = robot
        self.timestep = 32
        self.step = step
        self.goal = [0.195, 0.155]
        self.robot_state = state
        self.gps1 = gps1
        self.gps2 = gps2
        self.gps3 = gps3
        self.gps4 = gps4
        self.action = action
        if action == 0:    # 根据动作序号选择对应动作组，单位为rad
            self.ArmLower = 0
            self.Shoulder = 0.1
        else:
            self.ArmLower = 0.1
            self.Shoulder = 0
        self.if_jia = zhua
        self.jie1_Success = False
        self.motors = []
        self.motors_sensors = []
        self.motorName = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                     'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL', 'PelvR',
                     'PelvL', 'LegUpperR', 'LegUpperL', 'LegLowerR', 'LegLowerL',
                     'AnkleR', 'AnkleL', 'FootR', 'FootL', 'Neck', 'Head', 'GraspL', 'GraspR')
        for i in range(len(self.motorName)):   # 获取舵机和舵机传感器的设备对象
            self.motors.append(robot.getDevice(self.motorName[i]))
            sensorName = self.motorName[i]
            sensorName = sensorName + 'S'
            self.motors_sensors.append(self.robot.getDevice(sensorName))
            self.motors_sensors[i].enable(self.timestep)
        self.accelerometer = robot.getDevice('Accelerometer')
        self.gyro = robot.getDevice('Gyro')
        self.touch1 = self.robot.getDevice('touch_grasp_L1')    # 获取压力传感器的设备对象，当仿真环境中的压力传感器的碰撞体积与其他碰撞体积发生接触时，压力传感器会获得1的返回值，默认为0
        self.touch1_1 = self.robot.getDevice('touch_grasp_L1_1')
        self.touch1_2 = self.robot.getDevice('touch_grasp_L1_2')
        self.touch3 = self.robot.getDevice('touch_grasp_R1')
        self.touch3_1 = self.robot.getDevice('touch_grasp_R1_1')
        self.touch3_2 = self.robot.getDevice('touch_grasp_R1_2')
        self.touch5 = self.robot.getDevice('touch_foot_L1')
        self.touch6 = self.robot.getDevice('touch_foot_L2')
        self.touch7 = self.robot.getDevice('touch_foot_R1')
        self.touch8 = self.robot.getDevice('touch_foot_R2')
        self.touch11 = self.robot.getDevice('touch_arm_L1')
        self.touch12 = self.robot.getDevice('touch_arm_R1')
        self.touch13 = self.robot.getDevice('touch_leg_L1')
        self.touch14 = self.robot.getDevice('touch_leg_L2')
        self.touch15 = self.robot.getDevice('touch_leg_R1')
        self.touch16 = self.robot.getDevice('touch_leg_R2')
        self.touch1.enable(32)
        self.touch1_1.enable(32)
        self.touch1_2.enable(32)
        self.touch3.enable(32)
        self.touch3_1.enable(32)
        self.touch3_2.enable(32)
        self.touch5.enable(32)
        self.touch6.enable(32)
        self.touch7.enable(32)
        self.touch8.enable(32)
        self.touch11.enable(32)
        self.touch12.enable(32)
        self.touch13.enable(32)
        self.touch14.enable(32)
        self.touch15.enable(32)
        self.touch16.enable(32)
        self.touch = [self.touch1_2, self.touch3_2]   # 需要接触的压力传感器的列表
        self.touch_peng = [self.touch11, self.touch12, self.touch13, self.touch14, self.touch15, self.touch16]
        self.future_state = [i for i in self.robot_state]
        self.next = [self.robot_state[1] - self.Shoulder, self.robot_state[0] + self.Shoulder,
                     self.robot_state[5] + self.ArmLower, self.robot_state[4] - self.ArmLower]   # 根据动作计算出的各个舵机需要转动到的目标角度
        self.future_state[1] = self.next[0]
        self.future_state[0] = self.next[1]
        self.future_state[5] = self.next[2]
        self.future_state[4] = self.next[3]
        self.limit = [[-3.14, 3.14], [-3.14, 2.85], [-0.68, 2.3], [-2.25, 0.77], [-1.65, 1.16], [-1.18, 1.63],
                      [-2.42, 0.66], [-0.69, 2.5], [-1.01, 1.01], [-1, 0.93], [-1.77, 0.45], [-0.5, 1.68],
                      [-0.02, 2.25], [-2.25, 0.03], [-1.24, 1.38], [-1.39, 1.22], [-0.68, 1.04], [-1.02, 0.6],
                      [-1.81, 1.81], [-0.36, 0.94]]   # 每个舵机都有自己的运动范围，此为20个舵机的运动最小和最大限制角度
        self.now_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.next_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.touch_value = [0.0, 0.0]   # 用于记录压力传感器数值的列表
        self.touch_T = [1.0, 1.0]
        self.touch_F = [0.0, 0.0]
        self.acc_low = [480, 450, 580]   # 机器人运动过程中，加速度测量仪所允许的最小数值
        self.acc_high = [560, 530, 700]   # 机器人运动过程中，加速度测量仪所允许的最大数值
        self.gyro_low = [500, 500, 500]   # 机器人运动过程中，陀螺仪所允许的最小数值
        self.gyro_high = [520, 520, 520]   # 机器人运动过程中，陀螺仪所允许的最大数值

    def run(self):   # 运行函数
        self.robot.step(32)
        acc = self.accelerometer.getValues()
        gyro = self.gyro.getValues()
        # print("-------------------------")
        # print(f"acc: {acc}, gyro: {gyro}")
        x1 = self.goal[0] - self.gps1[1]
        x2 = self.goal[0] - self.gps2[1]
        y1 = self.goal[1] - self.gps1[2]
        y2 = self.goal[1] - self.gps2[2]
        goal = 0
        reward = 0
        reward1 = 20 - 200 * math.sqrt((x1 * x1) + (y1 * y1))
        reward2 = 20 - 200 * math.sqrt((x2 * x2) + (y2 * y2))
        count = 1
        for i in range(len(self.future_state)):
            if self.limit[1][0] <= self.future_state[i] <= self.limit[1][1]:
                continue
            else:
                reward = 0
                count = 0
                done = 1
                good = 1
                return self.next_state, reward, done, good, goal, count

        # print("acc:"+str(acc)+", gyro:"+str(gyro))

        for i in range(3):   # 检测机器人当前的加速度测量仪和陀螺仪，如果脱离了限定范围，则提前终止本轮训练，其中done是训练终止的标识符，good是合格样本数据的标识符，count是当前奖励需要进一步计算的标识符
            if self.acc_low[i] < acc[i] < self.acc_high[i] and self.gyro_low[i] < gyro[i] < self.gyro_high[i]:
                continue
            else:
                reward = 0
                count = 0
                done = 1
                good = 0
                return self.next_state, reward, done, good, goal, count
        if self.if_jia == 0.0:   # if_jia是控制机器人双手夹爪闭合的标识符，当其为0.0时，机器人正常根据指定动作运动；当其为1.0时，机器人则采用固定动作组，直接调用夹爪舵机，控制下夹爪闭合，尝试进行抓取动作
            self.motors[1].setPosition(self.next[0])
            self.motors[0].setPosition(self.next[1])
            self.motors[5].setPosition(self.next[2])
            self.motors[4].setPosition(self.next[3])
            self.robot.step(32)
            self.robot.step(32)
            self.robot.step(32)
            self.robot.step(32)
            self.robot.step(32)
            self.robot.step(32)
            self.robot.step(32)
            self.robot.step(32)
            done = 0
            reward = reward1 + reward2
            good = 1

            if self.touch1.getValue() == 1.0 or self.touch1_1.getValue() == 1.0 or self.touch1_2.getValue() == 1.0 \
                    or self.touch3 == 1.0 or self.touch3_1.getValue() == 1.0 or self.touch3_2.getValue() == 1.0:   # 检测夹爪内侧的压力传感器，如果有出现返回值，结束本轮训练，直接闭合双手夹爪
                timer = 0
                self.motors[21].setPosition(-0.5)
                self.motors[20].setPosition(-0.5)
                while self.robot.step(32) != -1:
                    timer += 32
                    if timer >= 2000:
                        break
                for j in range(len(self.touch)):   # 收集待测量压力传感器的数值
                    self.touch_value[j] = self.touch[j].getValue()
                sucess = np.array_equal(self.touch_value, self.touch_T)
                sucess = np.int(sucess)
                faild = np.array_equal(self.touch_value, self.touch_F)
                faild = np.int(faild)
                if faild == 1:
                    reward = 0
                    count = 1
                    done = 1
                    good = 1
                elif sucess == 1:
                    if (reward1 + reward2) < 20:
                        reward = 0
                        count = 1
                        done = 1
                        good = 1
                    else:
                        reward = 100
                        count = 0
                        done = 1
                        good = 1
                        print("俺抓到了")
                        goal = 1
                else:
                    if (reward1 + reward2) < 20:
                        count = 1
                        done = 1
                        good = 1
                    else:
                        count = 1
                        done = 1
                        good = 1
            else:
                for i in range(20):
                    self.next_state[i] = self.motors_sensors[i].getValue()
                    self.cha_zhi = self.next_state[i] - self.future_state[i]
                    if -0.005 < self.cha_zhi < 0.005:
                        continue
                    else:
                        count = 1
                        done = 1
                        good = 1
                        break
        else:
            timer = 0
            self.motors[21].setPosition(-0.5)
            self.motors[20].setPosition(-0.5)
            while self.robot.step(32) != -1:
                timer += 32
                if timer >= 2000:
                    break
            for j in range(len(self.touch)):
                self.touch_value[j] = self.touch[j].getValue()
            sucess = np.array_equal(self.touch_value, self.touch_T)
            sucess = np.int(sucess)
            faild = np.array_equal(self.touch_value, self.touch_F)
            faild = np.int(faild)
            if faild == 1 and self.step <= 5:
                reward = 0
                count = 1
                done = 1
                good = 1
            elif faild == 1 and self.step > 5:
                count = 1
                done = 1
                good = 1
            elif sucess == 1:
                if (reward1 + reward2) < 20:
                    reward = 0
                    count = 1
                    done = 1
                    good = 1
                else:
                    reward = 100
                    count = 0
                    done = 1
                    good = 1
                    print("俺抓到了")
                    goal = 1
            else:
                if (reward1 + reward2) < 20:
                    count = 1
                    done = 1
                    good = 1
                else:
                    count = 1
                    done = 1
                    good = 1
        return self.next_state, reward, done, good, goal, count