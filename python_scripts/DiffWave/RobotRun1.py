"""RobotRun controller."""
import math

#import gym
from re import X
import time
import numpy as np
import os
import sys
import cv2

import argparse
import platform

from pathlib import Path


from python_scripts.Project_config import path_list, gps_goal, gps_goal1

from python_scripts.Webots_interfaces import Darwin
from python_scripts.Project_config import Darwin_config



class RobotRun(Darwin):
    _prev_distance = None
    _prev_shoulder_action = 0.0
    _prev_arm_action = 0.0
    
    # 控制机器人按照action行动的类
    # action:
    def __init__(self, robot, discrete_action, continuous_action, step):
        super().__init__(robot)
        self.step = step    # 步数
        self.robot_state = self.get_robot_state()  # 机器人状态
        gps1, gps2, gps3, gps4, _ = self.get_gps_values()
        self.gps = [gps1, gps2, gps3, gps4]

        self.discrete_action = discrete_action
        self.continuous_action = continuous_action

        self.action_shouder, self.action_arm = self._map_policy_actions(
            self.discrete_action,
            self.continuous_action
        )

        # catch_flag：0=执行动作阶段，1=已触碰进入闭合验证阶段（由传感器动态设置）
        # 根据当前触碰传感器状态判断，不再用固定步数
        self.catch_flag = 1.0 if self._has_any_grasp_touch() else 0.0

        # 计算左臂和左肩的目标位置
        current_left_arm = self.robot_state[5]       # 左臂的当前状态在索引5
        current_left_shoulder = self.robot_state[1]  # 左肩的当前状态在索引1
        left_arm_target = 1.25 * self.action_arm + 0.25
        left_shoulder_target = 0.2995 * self.action_shouder - 0.145

        desired_arm_delta = max(-0.3, min(0.3, left_arm_target - current_left_arm))
        desired_shoulder_delta = max(-0.3, min(0.3, left_shoulder_target - current_left_shoulder))
        self.Shoulder = self._safe_symmetric_delta(1, 0, desired_shoulder_delta)
        self.ArmLower = self._safe_symmetric_delta(5, 4, desired_arm_delta)
        self.safety_clipped = (
            abs(self.Shoulder - desired_shoulder_delta) > 1e-6
            or abs(self.ArmLower - desired_arm_delta) > 1e-6
        )

        #print(f"手臂上升: {self.ArmLower}")
        # print(f"肩部上升: {self.Shoulder}")
        # print(f"手臂目标位置: {left_arm_target}")
        # print(f"肩部目标位置: {left_shoulder_target}")

        self.catch_Success_flag = False  # 抓取成功标识符
        #self.small_goal = 0  # 小目标
        # 初始化压力传感器列表
        self.touch = [self.touch_sensors['grasp_L1_1'], 
                      self.touch_sensors['grasp_R1_2']]
        # 压力传感器列表
        self.touch_peng = [self.touch_sensors['arm_L1'], 
                           self.touch_sensors['arm_R1'], 
                           self.touch_sensors['leg_L1'], 
                           self.touch_sensors['leg_L2'], 
                           self.touch_sensors['leg_R1'], 
                           self.touch_sensors['leg_R2']]
        self.future_state = [i for i in self.robot_state]  # 未来状态
        # 下一个状态
        self.next = [self.robot_state[1] + self.Shoulder,  #左肩
                     self.robot_state[0] - self.Shoulder,  #右肩
                     self.robot_state[5] + self.ArmLower,  #左臂
                     self.robot_state[4] - self.ArmLower]  #右臂
        # print(f"左肩: {self.robot_state[1]}")
        # print(f"右肩: {self.robot_state[0]}")
        # print(f"左臂: {self.robot_state[5]}")
        # print(f"右臂: {self.robot_state[4]}")
        self.future_state[1] = self.next[0]  # 未来状态[1] = 下一个状态[0]
        self.future_state[0] = self.next[1]  # 未来状态[0] = 下一个状态[1]
        self.future_state[5] = self.next[2]  # 未来状态[5] = 下一个状态[2]
        self.future_state[4] = self.next[3]  # 未来状态[4] = 下一个状态[3]

    
        self.now_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 当前状态
        self.next_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 下一个状态
        self.touch_value = [0.0, 0.0]  # 压力传感器值
        self.return_flag_list = {'reward': 0.0,
                     'done': 0,
                     'catch_success': False,
                     'catch_state': 'continue_search',  # 6种状态指标
                     'stair_score_1': 0.0,
                     'stair_score_2': 0.0,
                     'is_target_stair': 0,
                     'distance': 1.0}

    def _to_float(self, value, default=0.0):
        if value is None:
            return default
        if hasattr(value, 'item'):
            try:
                return float(value.item())
            except Exception:
                return default
        try:
            return float(value)
        except Exception:
            return default

    def _map_policy_actions(self, discrete_action, continuous_action):
        """根据离散动作决定连续舵机指令是否生效。"""
        d0 = self._to_float(discrete_action[0], 1.0) if len(discrete_action) > 0 else 1.0
        d1 = self._to_float(discrete_action[1], 1.0) if len(discrete_action) > 1 else 1.0

        cur_shoulder = self._to_float(continuous_action[0], RobotRun._prev_shoulder_action) if len(continuous_action) > 0 else RobotRun._prev_shoulder_action
        cur_arm = self._to_float(continuous_action[1], RobotRun._prev_arm_action) if len(continuous_action) > 1 else RobotRun._prev_arm_action

        mapped_shoulder = RobotRun._prev_shoulder_action if int(d0) == 0 else cur_shoulder
        mapped_arm = RobotRun._prev_arm_action if int(d1) == 0 else cur_arm

        RobotRun._prev_shoulder_action = mapped_shoulder
        RobotRun._prev_arm_action = mapped_arm
        return mapped_shoulder, mapped_arm

    def _safe_symmetric_delta(self, left_idx, right_idx, desired_delta):
        left_low, left_high = Darwin_config.limit[left_idx]
        right_low, right_high = Darwin_config.limit[right_idx]
        left_pos = self.robot_state[left_idx]
        right_pos = self.robot_state[right_idx]
        margin = 0.02
        lower = max(
            -0.3,
            left_low + margin - left_pos,
            right_pos - (right_high - margin),
        )
        upper = min(
            0.3,
            left_high - margin - left_pos,
            right_pos - (right_low + margin),
        )
        if lower > upper:
            lower = max(-0.3, left_low - left_pos, right_pos - right_high)
            upper = min(0.3, left_high - left_pos, right_pos - right_low)
        if lower > upper:
            return 0.0
        return float(np.clip(desired_delta, lower, upper))

    def _compute_distance_metrics(self):
        """计算两路GPS匹配评分，并给出是否命中目标台阶。"""
        try:
            gps1, gps2, gps3, gps4, _ = self.get_gps_values()
            self.gps = [gps1, gps2, gps3, gps4]
            y1 = Darwin_config.gps_goal[0] - self.gps[1][1]
            y2 = Darwin_config.gps_goal[0] - self.gps[2][1]
            z1 = Darwin_config.gps_goal[1] - self.gps[1][2]
            z2 = Darwin_config.gps_goal[1] - self.gps[2][2]
            stair_score_1 = 20 - 200 * math.sqrt((y1 * y1) + (z1 * z1))
            stair_score_2 = 20 - 200 * math.sqrt((y2 * y2) + (z2 * z2))
            distance = math.sqrt((gps_goal1[0] - self.gps[0][1]) ** 2 + (gps_goal1[1] - self.gps[0][2]) ** 2)
        except Exception:
            stair_score_1, stair_score_2, distance = -20.0, -20.0, 1.0

        is_target_stair = int(distance <= 0.15)
        return stair_score_1, stair_score_2, distance, is_target_stair

    def _check_joint_limits(self):
        """关节角度约束：future_state每个关节都应在限制范围内。"""
        for i in range(min(len(self.future_state), len(Darwin_config.limit))):
            low, high = Darwin_config.limit[i]
            if not (low <= self.future_state[i] <= high):
                return False
        return True

    def _check_imu_limits(self):
        """加速度和陀螺仪约束。"""
        acc = self.accelerometer.getValues()
        gyro = self.gyro.getValues()
        for i in range(3):
            if not (Darwin_config.acc_low[i] < acc[i] < Darwin_config.acc_high[i] and
                    Darwin_config.gyro_low[i] < gyro[i] < Darwin_config.gyro_high[i]):
                return False
        return True

    def _apply_upper_body_action(self):
        """执行抓取阶段的四个上肢关节动作。"""
        self.motors[1].setPosition(self.next[0])
        self.motors[0].setPosition(self.next[1])
        self.motors[5].setPosition(self.next[2])
        self.motors[4].setPosition(self.next[3])

    def _wait_action_settle(self, max_steps=100, tol=0.01):
        """等待关键关节到达目标，防止动作尚未执行完成就进入判定。"""
        target_positions = [self.next[0], self.next[1], self.next[2], self.next[3]]
        sensor_indices = [1, 0, 5, 4]
        for _ in range(max_steps):
            if self.robot.step(self.timestep) == -1:
                break
            current_positions = [self.motors_sensors[idx].getValue() for idx in sensor_indices]
            if all(abs(t - c) <= tol for t, c in zip(target_positions, current_positions)):
                return True
        return False

    def _has_any_grasp_touch(self):
        """抓爪任一触碰传感器触发即返回True。"""
        sensors = [
            'grasp_L1', 'grasp_L1_1', 'grasp_L1_2',
            'grasp_R1', 'grasp_R1_1', 'grasp_R1_2'
        ]
        return any(self.touch_sensors[name].getValue() == 1.0 for name in sensors)

    def _close_grasp_and_read_pair(self, wait_ms=2000):
        """闭合抓爪并读取老版本使用的两路触碰模板。"""
        self.motors[21].setPosition(-0.5)
        self.motors[20].setPosition(-0.5)
        timer = 0
        while self.robot.step(self.timestep) != -1:
            timer += self.timestep
            if timer >= wait_ms:
                break

        for j in range(len(self.touch)):
            self.touch_value[j] = self.touch[j].getValue()

        all_grasp_sensors = [
            self.touch_sensors['grasp_L1'].getValue(),
            self.touch_sensors['grasp_L1_1'].getValue(),
            self.touch_sensors['grasp_L1_2'].getValue(),
            self.touch_sensors['grasp_R1'].getValue(),
            self.touch_sensors['grasp_R1_1'].getValue(),
            self.touch_sensors['grasp_R1_2'].getValue(),
        ]
        left_any = any(all_grasp_sensors[0:3])
        right_any = any(all_grasp_sensors[3:6])
        strict_success = np.array_equal(self.touch_value, Darwin_config.touch_T)
        relaxed_success = left_any and right_any
        success = int(strict_success or relaxed_success)
        failed = int(np.array_equal(self.touch_value, Darwin_config.touch_F) and not left_any and not right_any)
        print(f"抓取判断 - 传感器值: {self.touch_value}, 所有传感器: {all_grasp_sensors}")
        print(f"左侧触发: {left_any}, 右侧触发: {right_any}, 成功: {success}, 失败: {failed}")
        return success, failed

    def _check_collision_constraint(self):
        """碰撞约束：手臂与腿部触碰传感器不应触发。"""
        return any(sensor.getValue() == 1.0 for sensor in self.touch_peng)

    def _refresh_next_state(self):
        """更新next_state，供训练器作为下一时刻状态。"""
        state = self.get_robot_state()
        for i in range(min(20, len(state))):
            self.next_state[i] = state[i]

    def _check_tracking_constraint(self, tol=0.005):
        """执行后跟踪约束：关节实测值与future_state偏差不应过大。"""
        for i in (0, 1, 4, 5):
            self.next_state[i] = self.motors_sensors[i].getValue()
            if abs(self.next_state[i] - self.future_state[i]) > tol:
                return False
        return True

    def _log_constraint_trigger(self, name, details=""):
        """统一打印约束触发日志，便于训练时快速定位终止原因。"""
        msg = f"[Constraint Triggered] step={self.step} | {name}"
        if details:
            msg += f" | {details}"
        print(msg)

    def compute_reward(self, distance, prev_distance, done, success, failed, goal, collision, imu_ok, joints_ok):
        """统一奖励计算：融合距离变化、接近奖励、动作惩罚与抓取终局奖励。"""
        if not joints_ok:
            return 0.0
        if not imu_ok:
            return 0.0

        # 1) 距离变化奖励（鼓励朝目标逼近）
        if prev_distance is not None:
            distance_reward = (prev_distance - distance) * 15.0
        else:
            distance_reward = -distance

        # 2) 接近奖励
        proximity_reward = max(0.0, (0.5 - distance) * 5.0)

        # 3) 动作惩罚
        action_magnitude = (abs(float(self.action_shouder)) + abs(float(self.action_arm))) / 2.0
        inactivity_penalty = -0.2 if action_magnitude < 0.05 else 0.0
        large_action_penalty = -0.05 if (abs(float(self.action_shouder)) > 0.9 or abs(float(self.action_arm)) > 0.9) else 0.0

        reward = distance_reward + proximity_reward + inactivity_penalty + large_action_penalty

        # 4) 时间惩罚
        reward -= 0.5 * float(self.step)

        # 硬约束相关
        if collision:
            reward -= 8.0

        # 5) 抓取终局奖励/惩罚
        if done == 1:
            if success == 1:
                if goal == 1:
                    reward += 30.0
                else:
                    # 抓到但不是目标台阶，给较低等级奖励而不是大奖励
                    reward += 8.0
            elif failed == 1:
                reward -= 5.0
            else:
                reward -= 10.0

        return reward

    def run(self):
        self.robot.step(self.timestep)

        if int(self.step) <= 0:
            RobotRun._prev_distance = None
            RobotRun._prev_shoulder_action = 0.0
            RobotRun._prev_arm_action = 0.0

        # 默认返回值
        self.return_flag_list.update({'reward': 0.0, 'done': 0, 'catch_success': False, 'catch_state': 'continue_search'})
        goal_flag = 0

        stair_score_1, stair_score_2, distance, is_target_stair = self._compute_distance_metrics()
        self.return_flag_list.update({'stair_score_1': stair_score_1, 'stair_score_2': stair_score_2, 
                                      'is_target_stair': is_target_stair, 'distance': distance})
        joints_ok = self._check_joint_limits()
        imu_ok = self._check_imu_limits()
        
        # 打印step开始信息
        print(f"\n[Step {int(self.step)}] d_action={int(self.discrete_action[0])},{int(self.discrete_action[1])} | c_action={self.action_shouder:.3f},{self.action_arm:.3f} | distance={distance:.4f}")
        if self.safety_clipped:
            print(
                "  安全过滤(RobotRun): "
                f"shoulder_delta={self.Shoulder:.4f}, arm_delta={self.ArmLower:.4f}"
            )

        # 约束1：关节角度限制
        if not joints_ok:
            self._log_constraint_trigger('joint_limits', 'future_state out of Darwin_config.limit')
            self.return_flag_list.update({'reward': 0.0, 'done': 1, 'catch_success': False, 'catch_state': 'terminated_by_constraint'})
            self._refresh_next_state()
            return self.next_state, \
                   self.return_flag_list['reward'], \
                   self.return_flag_list['done'], \
                 self.return_flag_list['catch_success']

        # 约束2：IMU传感器范围
        if not imu_ok:
            self._log_constraint_trigger('imu_limits', 'accelerometer/gyro out of threshold')
            self.return_flag_list.update({'reward': 0.0, 'done': 1, 'catch_success': False, 'catch_state': 'terminated_by_constraint'})
            self._refresh_next_state()
            return self.next_state, \
                   self.return_flag_list['reward'], \
                   self.return_flag_list['done'], \
                 self.return_flag_list['catch_success']

        success = 0
        failed = 0
        collision = False

        # catch_flag=0：执行抓取动作阶段
        if float(self.catch_flag) == 0.0:
            self._apply_upper_body_action()
            self._wait_action_settle(max_steps=100, tol=0.01)

            # 约束3：碰撞约束
            collision = self._check_collision_constraint()
            if collision:
                self._log_constraint_trigger('collision', 'arm/leg touch sensors are active')
                self.return_flag_list.update({'done': 1, 'catch_success': False, 'catch_state': 'terminated_by_constraint'})

            # 约束4：抓爪触碰触发后闭合并判定成功/失败
            elif self._has_any_grasp_touch():
                self._log_constraint_trigger('grasp_touch', 'grasp sensor touched, start closing')
                # 已触碰，进入闭合验证阶段
                self.catch_flag = 1.0
                self.return_flag_list['catch_state'] = 'attempt_now'
                success, failed = self._close_grasp_and_read_pair(wait_ms=2000)
                self.return_flag_list.update({'done': 1})
                if success == 1:
                    self._log_constraint_trigger('grasp_result', 'close success')
                    self.catch_Success_flag = True
                    # 抓取动作执行后再判定是否命中目标台阶
                    stair_score_1, stair_score_2, distance, is_target_stair = self._compute_distance_metrics()
                    self.return_flag_list.update({'stair_score_1': stair_score_1, 'stair_score_2': stair_score_2, 
                                                  'is_target_stair': is_target_stair, 'distance': distance})
                    if is_target_stair == 1:
                        self._log_constraint_trigger('target_check', 'grasped_right')
                        goal_flag = 1
                        self.return_flag_list['catch_success'] = True
                        self.return_flag_list['catch_state'] = 'grasped_right'
                    else:
                        self._log_constraint_trigger('target_check', 'grasped_wrong')
                        # 抓到但不是目标台阶
                        self.return_flag_list['catch_state'] = 'grasped_wrong'
                elif failed == 1:
                    self._log_constraint_trigger('grasp_result', 'close failed')
                    self.return_flag_list['catch_state'] = 'attempt_failed'
                    goal_flag = 0

            # 约束5：执行后跟踪偏差约束
            else:
                tracking_ok = self._check_tracking_constraint(tol=0.02)
                if not tracking_ok:
                    self._log_constraint_trigger('tracking_error', 'joint tracking deviation > 0.02')
                    self.return_flag_list.update({'done': 1, 'catch_success': False, 'catch_state': 'terminated_by_constraint'})
                # 如果都未触碰且约束通过，保留 catch_state='continue_search'

        # catch_flag=1：已触碰，保持闭合并继续验证
        else:
            self._log_constraint_trigger('grasp_touch', 'carry-over close validation branch')
            success, failed = self._close_grasp_and_read_pair(wait_ms=2000)
            self.return_flag_list.update({'done': 1})
            # 同样在闭合动作后重新判断台阶匹配
            stair_score_1, stair_score_2, distance, is_target_stair = self._compute_distance_metrics()
            self.return_flag_list.update({'stair_score_1': stair_score_1, 'stair_score_2': stair_score_2, 
                                          'is_target_stair': is_target_stair, 'distance': distance})
            if success == 1 and is_target_stair == 1:
                self._log_constraint_trigger('target_check', 'grasped_right')
                goal_flag = 1
                self.return_flag_list['catch_success'] = True
                self.return_flag_list['catch_state'] = 'grasped_right'
            elif success == 1:
                self._log_constraint_trigger('target_check', 'grasped_wrong')
                self.return_flag_list['catch_state'] = 'grasped_wrong'
            elif failed == 1:
                self._log_constraint_trigger('grasp_result', 'close failed')
                self.return_flag_list['catch_state'] = 'attempt_failed'

        # 奖励计算前刷新一次距离，确保本步reward使用动作后的状态
        stair_score_1, stair_score_2, distance, _ = self._compute_distance_metrics()

        # 统一奖励计算
        self.return_flag_list['reward'] = self.compute_reward(
            distance=distance,
            prev_distance=RobotRun._prev_distance,
            done=self.return_flag_list['done'],
            success=success,
            failed=failed,
            goal=goal_flag,
            collision=collision,
            imu_ok=imu_ok,
            joints_ok=joints_ok
        )

        # 更新跨step的距离记忆，用于下一步距离变化奖励
        RobotRun._prev_distance = distance
        if self.return_flag_list['done'] == 1:
            RobotRun._prev_distance = None

        # 确保next_state在返回前已更新
        self._refresh_next_state()
        
        # 打印step结束信息
        print(f"  → Result: reward={self.return_flag_list['reward']:.2f} | done={self.return_flag_list['done']} | catch_state={self.return_flag_list['catch_state']} | catch_success={self.return_flag_list['catch_success']}")

        return self.next_state, \
               self.return_flag_list['reward'], \
               self.return_flag_list['done'], \
             self.return_flag_list['catch_success']
    
