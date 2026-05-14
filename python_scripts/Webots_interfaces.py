import os
import sys
from controller import Robot, Motor, Motion, LED, Camera, Gyro, Accelerometer, PositionSensor, GPS
from PIL import Image
import numpy as np
import torch
import shutil
sys.path.append('D:/Webots/Webots2021a/Webots/projects/robots/robotis/darwin-op/libraries/python37')
#from managers import RobotisOp2GaitManager, RobotisOp2MotionManager
from python_scripts.Project_config import path_list

class Darwin:
    """Darwin机器人类
    
    功能：封装Darwin机器人的所有操作接口
    
    属性：
        motors: 电机设备列表
        sensors: 传感器设备列表
        camera: 相机设备
        gps: GPS设备组
        touch_sensors: 触碰传感器设备列表
        其他传感器设备...
    """
    def __init__(self, robot):
        # 基础设备初始化
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        #self.gaitManager = RobotisOp2GaitManager(robot, 'config.ini')
        #self.motionManager = RobotisOp2MotionManager(robot)
        #self.gaitManager.setBalanceEnable(True)
        # 舵机列表初始化
        self.motors = []
        self.motors_sensors = []
        self.motorName = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                          'ArmLowerR', 'ArmLowerL', 'PelvYR',    'PelvYL', 
                          'PelvR',     'PelvL',     'LegUpperR', 'LegUpperL', 
                          'LegLowerR', 'LegLowerL', 'AnkleR',    'AnkleL', 
                          'FootR',     'FootL',     'Neck',      'Head', 
                          'GraspL',    'GraspR')
        self._init_motors()
        # LED设备
        self.eyeLed = robot.getDevice('EyeLed')
        self.headLed = robot.getDevice('HeadLed')
        # 传感器设备
        self.camera = robot.getDevice('Camera') # 摄像头
        self._camera_debug_printed = False
        self.accelerometer = robot.getDevice('Accelerometer')    # 加速度传感器
        self.gyro = robot.getDevice('Gyro')    # 陀螺仪传感器
        # GPS设备组
        self.left_gps1 = robot.getDevice('left_gps1')
        self.right_gps1 = robot.getDevice('right_gps1')
        self.left_gps2 = robot.getDevice('left_gps2')
        self.right_gps2 = robot.getDevice('right_gps2')
        self.foot_gps1 = robot.getDevice('foot_gps1')
        # 启用传感器
        self.enable_sensors()
        # 初始化触碰传感器(字典格式{传感器名称:传感器数值})
        self.touch_sensors = {}
        self._init_touch_sensors() 

    def enable_sensors(self):
        """启用所有传感器"""
        self.camera.enable(self.timestep)
        self.accelerometer.enable(self.timestep)
        self.gyro.enable(self.timestep)
        self.left_gps1.enable(self.timestep)
        self.right_gps1.enable(self.timestep)
        self.left_gps2.enable(self.timestep)
        self.right_gps2.enable(self.timestep)
        self.foot_gps1.enable(self.timestep)

    def _init_touch_sensors(self):
        """初始化触碰传感器"""
        # 定义所有触碰传感器(字典格式{传感器名称:传感器名称})
        self.sensors_config = {
            'grasp_L1'  : 'touch_grasp_L1',
            'grasp_L1_1': 'touch_grasp_L1_1',
            'grasp_L1_2': 'touch_grasp_L1_2',
            'grasp_R1'  : 'touch_grasp_R1',
            'grasp_R1_1': 'touch_grasp_R1_1',
            'grasp_R1_2': 'touch_grasp_R1_2',
            'foot_L1'   : 'touch_foot_L1',
            'foot_L2'   : 'touch_foot_L2',
            'foot_R1'   : 'touch_foot_R1',
            'foot_R2'   : 'touch_foot_R2',
            'arm_L1'    : 'touch_arm_L1',
            'arm_R1'    : 'touch_arm_R1',
            'leg_L1'    : 'touch_leg_L1',
            'leg_L2'    : 'touch_leg_L2',
            'leg_R1'    : 'touch_leg_R1',
            'leg_R2'    : 'touch_leg_R2'
        }
        
        # 初始化并启用每个触碰传感器
        for key, name in self.sensors_config.items():
            sensor = self.robot.getDevice(name)
            sensor.enable(self.timestep)
            self.touch_sensors[key] = sensor
            # print(f'self.touch_sensors_{key}=', self.touch_sensors[key].getValue())

    def _init_motors(self):
        """初始化所有电机和位置传感器"""
        for name in self.motorName:
            motor = self.robot.getDevice(name)
            # 电机不需要enable，创建后即可使用
            self.motors.append(motor)
            # 初始化对应的位置传感器
            sensor = self.robot.getDevice(name + 'S')
            sensor.enable(self.timestep)
            self.motors_sensors.append(sensor)

    def robot_reset(self):
        """重置机器人到初始姿态"""
        a = -np.random.random() * 0.4 + 0.7
        b = -np.random.random() * 0.4 + 1.0
        
        # 设置基础姿态
        self._set_initial_pose(a, b)
        # 设置电机速度
        self._set_motors_velocity()
        # 等待稳定
        for _ in range(24):
            self.robot.step(self.timestep)
    

    def _set_initial_pose(self, a, b):
        """设置机器人初始姿态
        
        参数：
            (a,b are defined in robot_reset())
            a: 肩部角度随机值
            b: 手臂角度随机值
        """
        pose_config = {
            'GraspL': 1, 'GraspR': 1,
            'Neck': 0.2, 'Head': 0,
            'PelvYL': 0.0, 'PelvL': 0.0,
            'LegUpperL': 0.5, 'LegLowerL': -0.2,
            'AnkleL': 0, 'FootL': 0.0,
            'PelvYR': 0.0, 'PelvR': 0.0,
            'LegUpperR': -0.5, 'LegLowerR': 0.2,
            'AnkleR': 0, 'FootR': 0.0,
            'ShoulderL': a, 'ArmUpperL': 0.67, 'ArmLowerL': -b,
            'ShoulderR': -a, 'ArmUpperR': -0.67, 'ArmLowerR': b
        }
        
        for name, position in pose_config.items():
            motor_idx = self.motorName.index(name)
            self.motors[motor_idx].setPosition(position)



    def _set_motors_velocity(self, velocity=1.0):
        """设置所有电机的速度
        
        参数：
            velocity: 目标速度值(原来文件中设置的速度都是1.0)
        """
        for motor in self.motors:
            motor.setVelocity(velocity)

    def get_gps_values(self):
        """获取所有GPS数据
        
        返回：
            tuple: 包含五个GPS数据
        """
        return (
            self.left_gps1.getValues(),
            self.right_gps1.getValues(),
            self.left_gps2.getValues(),
            self.right_gps2.getValues(),
            self.foot_gps1.getValues()
        )
    
    def get_touch_values(self):
        """获取所有触碰传感器的值
        
        返回：
            dict: 包含所有触碰传感器的状态的字典
                键: 传感器名称 (如 'grasp_L1', 'foot_L1' 等)
                值: 传感器读数
                    - 1.0: 传感器被触碰
                    - 0.0: 传感器未被触碰
        """
        return {key: sensor.getValue() for key, sensor in self.touch_sensors.items()}

    def get_touch_sensor_value(self, sensor_name):
        """获取指定触碰传感器的值
        参数:
            sensor_name: 传感器名称，如 'grasp_L1_2'
        返回:
            float: 传感器值，1.0表示被触碰，0.0表示未被触碰
        """
        if sensor_name in self.touch_sensors:
            return self.touch_sensors[sensor_name].getValue()
        return 0.0

    def lock_grasp(self):
        """锁定夹爪位置，确保夹爪保持闭合状态"""
        # 设置夹爪电机位置为闭合状态
        self.motors[20].setPosition(-0.5)
        self.motors[21].setPosition(-0.5)
      
        # 等待夹爪动作完成
        for _ in range(10):
            self.robot.step(self.timestep)

    
    def check_grasp_contact(self):
        """检查抓取器的接触状态（左右各有三个抓取传感器）
        
        返回：
            dict: 包含左右两侧抓取器接触状态的字典
                键: 'left' 或 'right'
                值: bool, 是否接触
        """
        left_grasp = any([
            self.touch_sensors['grasp_L1'].getValue(),
            self.touch_sensors['grasp_L1_1'].getValue(),
            self.touch_sensors['grasp_L1_2'].getValue()
        ])
        
        right_grasp = any([
            self.touch_sensors['grasp_R1'].getValue(),
            self.touch_sensors['grasp_R1_1'].getValue(),
            self.touch_sensors['grasp_R1_2'].getValue()
        ])
        
        return {'left': left_grasp, 'right': right_grasp}    
    #这里是从touch_peng中提取的
    def check_collision(self):
        """检查是否发生碰撞（通过检查手臂和腿部的触碰传感器）
        
        返回：
            bool: 是否发生碰撞
        """
        collision_sensors = [
            'arm_L1', 'arm_R1',
            'leg_L1', 'leg_L2',
            'leg_R1', 'leg_R2'
        ]
        
        return any(self.touch_sensors[sensor].getValue() for sensor in collision_sensors)    

    def get_camera_image(self, step):
        width = int(self.camera.getWidth())
        height = int(self.camera.getHeight())
        raw = self.camera.getImage()
        if raw is None:
            raise RuntimeError("Camera buffer is empty; cannot provide a fresh image to the policy.")

        expected_size = width * height * 4
        raw_array = np.frombuffer(raw, dtype=np.uint8)
        if raw_array.size != expected_size:
            raise RuntimeError(
                "Camera buffer size mismatch: expected %d bytes for %dx%d BGRA, got %d."
                % (expected_size, width, height, raw_array.size)
            )

        bgra = raw_array.reshape((height, width, 4))
        rgb = bgra[:, :, 2::-1]
        gray = (
            0.299 * rgb[:, :, 0].astype(np.float32)
            + 0.587 * rgb[:, :, 1].astype(np.float32)
            + 0.114 * rgb[:, :, 2].astype(np.float32)
        ).astype(np.uint8)

        if width != 128 or height != 128:
            gray = np.array(Image.fromarray(gray).resize((128, 128), Image.BILINEAR), dtype=np.uint8)

        img_array = gray.astype(np.float32) / 255.0
        img_name = f"img{int(step)}.png"
        img_path = os.path.abspath(img_name)
        try:
            Image.fromarray(gray).save(img_path)
            os.utime(img_path, None)
            img_mtime = os.path.getmtime(img_path)
        except Exception as exc:
            print(f"Warning: failed to save camera image {img_path}: {exc}")
            img_mtime = None

        if not self._camera_debug_printed:
            print(
                "Camera image source: cwd=%s, size=%dx%d, saved_file=%s, mtime=%s, min=%.4f, max=%.4f, mean=%.4f"
                % (
                    os.getcwd(),
                    width,
                    height,
                    img_path,
                    img_mtime,
                    float(np.min(img_array)),
                    float(np.max(img_array)),
                    float(np.mean(img_array)),
                )
            )
            self._camera_debug_printed = True

        img_tensor = torch.tensor(img_array).unsqueeze(0).float()
        return img_array, img_tensor

    def get_robot_state(self):
        """获取机器人关节状态
        
        返回：
            list: 包含所有关节位置的列表
        """
        # robot_state = []
        # print(f'motors_sensors: {self.motors_sensors}')
        # print(f'motors_sensors_len: {len(self.motors_sensors)}')
        # for sensor in self.motors_sensors[:-2]:
            # print(f'sensor: {sensor}')
            # print(f'sensor.getValue(): {sensor.getValue()}')
            # robot_state.append(sensor.getValue())
        robot_state = [sensor.getValue() for sensor in self.motors_sensors[:-2]]
        # print(f'webots_interfaces: robot_state: {robot_state}')
        # print(f'webots_interfaces: robot_state_len: {len(robot_state)}')
        return robot_state

    def check_acceleration_and_gyro(self):
        """检查加速度和陀螺仪数据是否在正常范围内
        
        返回：
            bool: 是否在正常范围内
        """
        acc = self.accelerometer.getValues()
        gyro = self.gyro.getValues()
        #limits 是参考RobotRun2.py和RobotRun1.py的
        acc_limits = {
            'low': [480, 450, 580],
            'high': [560, 530, 700]
        }
        gyro_limits = {
            'low': [500, 500, 500],
            'high': [520, 520, 520]
        }
        
        # 检查加速度
        for i in range(3):
            if not (acc_limits['low'][i] < acc[i] < acc_limits['high'][i]):
                return False
                
        # 检查陀螺仪
        for i in range(3):
            if not (gyro_limits['low'][i] < gyro[i] < gyro_limits['high'][i]):
                return False
                
        return True

    def check_joint_limits(self, positions):
        """检查关节位置是否在限制范围内
        
        参数：
            positions: list, 关节位置列表
        返回：
            bool: 是否所有关节都在限制范围内
        """
        #limits 依然是参考RobotRun2.py和RobotRun1.py的
        limits = [
            [-3.14, 3.14], [-3.14, 2.85], [-0.68, 2.3], 
            [-2.25, 0.77], [-1.65, 1.16], [-1.18, 1.63],
            [-2.42, 0.66], [-0.69, 2.5], [-1.01, 1.01], 
            [-1, 0.93], [-1.77, 0.45], [-0.5, 1.68],
            [-0.02, 2.25], [-2.25, 0.03], [-1.24, 1.38], 
            [-1.39, 1.22], [-0.68, 1.04], [-1.02, 0.6],
            [-1.81, 1.81], [-0.36, 0.94]
        ]
        
        for i in range(len(positions)):
            pos = positions[i]
            if not (limits[i][0] <= pos <= limits[i][1]):
                return False
        return True
    #===============================================以下是固定动作组函数===============================================
    def execute_timed_motion(self, motor_positions, duration, velocity=1.0):
        """执行定时动作
        参数：
            motor_positions: dict, 电机名称和目标位置的映射
            duration: int, 动作持续时间(ms)
            velocity: float, 电机速度{默认1.0}
        """
        timer = 0
        for name, position in motor_positions.items():
            motor = self.motors[self.motorName.index(name)]
            motor.setPosition(position)
            motor.setVelocity(velocity)
        
        while self.robot.step(32) != -1:
            timer += 32
            if timer >= duration:
                break

    def _set_left_leg_initpose(self):
        """设置左腿初始姿态"""
        self.execute_timed_motion({
            'LegUpperL': 1.65, 
            'LegLowerL': -2.2,
            'AnkleL': -0.85
        }, 1500, 1.5)  
        self.execute_timed_motion({
            'LegUpperL': 0.4, 
            'LegLowerL': -0.7,
            'AnkleL': -0.7
        }, 1500, 1.0)
        self.execute_timed_motion({
            'LegLowerL': -0.1,
            'AnkleL': -0.15
        }, 1500, 1.0)

    def tai_leg_L1(self):
        """抬起左腿第一阶段"""
        self.execute_timed_motion({
            'LegLowerL': -0.7,
            'AnkleL': -0.5
        }, 2000)

    def tai_leg_L2(self):
        """抬起左腿第二阶段"""
        self.execute_timed_motion({
            'LegUpperL': 1.65,
            'LegLowerL': -2.2,
            'AnkleL': -0.85
        }, 2000, 2)

    def tai_leg_L3(self):
        """抬起左腿第三阶段"""
        self.execute_timed_motion({
            'LegLowerL': -1.8,
            'AnkleL': -0.45
        }, 1000)

    def tai_leg_L4(self):
        """抬起左腿第四阶段"""
        self.execute_timed_motion({
            'LegUpperL': 1.4,
            'LegLowerL': -1.55,
            'AnkleL': -0.45
        }, 1000)

    def n_tai_leg_L1(self):
        """新抬起左腿第一阶段"""
        self.execute_timed_motion({
            'LegLowerL': -1.45,
            'AnkleL': -0.45
        }, 2000)

    def n_tai_leg_L2(self):
        """新抬起左腿第二阶段"""
        self.execute_timed_motion({
            'LegUpperL': 1.65,
            'LegLowerL': -2.2,
            'AnkleL': -0.85
        }, 2000, 2)

    def n_tai_leg_L3(self):
        """新抬起左腿第三阶段"""
        self.execute_timed_motion({
            'LegLowerL': -1.8,
            'AnkleL': -0.8,
            'AnkleR': -0.8,
            'ArmLowerL': 0.2,
            'ShoulderL': -0.4,
            'ArmLowerR': -0.2,
            'ShoulderR': 0.4
        }, 2000)

    def n_tai_leg_L4(self):
        """新抬起左腿第四阶段"""
        self.execute_timed_motion({
            'LegLowerL': -1.6,
            'AnkleL': -0.45
        }, 1000)

    def tai_leg_R1(self):
        """抬起右腿第一阶段"""
        self.execute_timed_motion({
            'LegLowerR': 0.7,
            'AnkleR': 0.35
        }, 2000)

    def tai_leg_R2(self):
        """抬起右腿第二阶段"""
        self.execute_timed_motion({
            'LegUpperR': -1.65,
            'LegLowerR': 2.2,
            'AnkleR': 0.85
        }, 2000, 2)

    def tai_leg_R3(self):
        """抬起右腿第三阶段"""
        self.execute_timed_motion({
            'LegLowerR': 1.8,
            'AnkleR': 0.45
        }, 1000)

    def tai_leg_R4(self):
        """抬起右腿第四阶段"""
        self.execute_timed_motion({
            'LegUpperR': -1.4,
            'LegLowerR': 1.55,
            'AnkleR': 0.45
        }, 1000)

    def qi_li(self):
        """起立动作"""
        self.execute_timed_motion({
            'LegLowerL': -1.1,
            'LegUpperL': 1.45,
            'LegLowerR': 1.1,
            'LegUpperR': -1.45,
            'ArmLowerL': -0.1,
            'ArmLowerR': 0.1,
            'AnkleL': -0.25,
            'AnkleR': 0.25,
            'ShoulderL': -0.05,
            'ShoulderR': 0.05,
            'GraspL': 0,
            'GraspR': 0
        }, 2000)

    def song_L(self):
        """松开左手"""
        self.execute_timed_motion({
            'GraspL': 1
        }, 1000, 2)

    def song_R(self):
        """松开右手"""
        self.execute_timed_motion({
            'GraspR': 1
        }, 1000, 2)

    # ... 后续可以自己定义

class Environment:
    """基础环境类
    功能：管理仿真环境的基本操作
    属性：
        robot: Webots机器人实例
        timestep: 仿真步长
    """

    def __init__(self, robot=None):
        if robot is None:
            self.robot = Robot()
        else:
            self.robot = robot
        self.timestep = int(self.robot.getBasicTimeStep())
        self.darwin = Darwin(self.robot)  # Darwin实例
        self.state = None
        self.done = False
        self.isSuccess = False

    def myStep(self):
        """推进仿真一个时间步长"""
        return self.robot.step(self.timestep)

    def wait(self, ms):
        """等待指定时间
        参数：
            ms: 等待毫秒数
        """
        startTime = self.robot.getTime()
        s = ms / 1000
        while s + startTime >= self.robot.getTime():
            self.myStep()
    def wait_until_stable(self, max_retry=60, verbose=True):
        for i in range(max_retry):
            self.robot.step(self.timestep)
            if self.darwin.check_acceleration_and_gyro():
                if verbose:
                    print(f"传感器已稳定，重试次数: {i + 1}/{max_retry}")
                return True
            if verbose:
                print(f"等待传感器稳定... ({i + 1}/{max_retry})")
        if verbose:
            print(f"警告: 达到最大重试次数({max_retry})，可能被卡住")
        return False
    def reset(self):
        self.darwin.robot_reset()

        ok = self.wait_until_stable(max_retry=60)
        self.darwin.robot_reset()

        ok = self.wait_until_stable(max_retry=60, verbose=True)
        if not ok:
            print("警告: reset 后传感器仍未稳定")

        with open(path_list['resetFlag'], 'r+') as file:
            file.write('0')
        with open(path_list['resetFlag1'], 'r+') as file:
            file.write('0')

        self.done = False
        return self.state

    
    def step(self, state, action_shouder, action_arm, steps, catch_flag, gps1, gps2, gps3, gps4, img_name):
        """执行一步动作
        参数:
            state: 当前状态
            action_shouder: 肩膀舵机动作
            action_arm: 手臂舵机动作
            steps: 步数
            catch_flag: 抓取器状态
            gps1-4: GPS位置信息
            name: 动作名称
        返回:
            tuple: (next_state, reward, done, good, goal, count)
        """
        if not self.darwin.check_acceleration_and_gyro():
            print("step前传感器不稳定，等待恢复...")
            stable = self.wait_until_stable(max_retry=20, verbose=True)
            if not stable:
                print("step前仍不稳定，直接结束本回合")
                next_state = self.get_robot_state()
                return next_state, 0, 1, 0, 0, 0
        from python_scripts.PPO.RobotRun1 import RobotRun
        import inspect

        param_count = len(inspect.signature(RobotRun.__init__).parameters)
        if param_count == 5:
            next_state, reward, done, catch_success = RobotRun(
                self.robot,
                [1, 1],
                [float(action_shouder), float(action_arm)],
                steps
            ).run()
            good = 1
            goal = 1 if catch_success else 0
            count = 0 if catch_success else 1
            return next_state, reward, done, good, goal, count

        return RobotRun(self.robot, state, action_shouder, action_arm, steps, catch_flag, gps1, gps2, gps3, gps4, img_name).run()
    
    def step2(self, state, action_leg_upper, action_leg_lower, action_ankle, steps, zhua, gps0, gps1, gps2, gps3, gps4):
        if not self.darwin.check_acceleration_and_gyro():
            print("step2前传感器不稳定，等待恢复...")
            stable = self.wait_until_stable(max_retry=20, verbose=True)
            if not stable:
                print("step2前仍不稳定，直接结束本回合")
                next_state = self.get_robot_state()
                return next_state, 0, 1, 0, 0, 0
        from python_scripts.PPO.RobotRun2 import RobotRun2
        return RobotRun2(self.robot, state, action_leg_upper, action_leg_lower, action_ankle, steps, zhua, gps0, gps1, gps2, gps3, gps4).run()
    
    def get_robot_state(self):
        """获取机器人的关节状态，即舵机角度"""
        return self.darwin.get_robot_state()
    
    def get_img(self, step, imgs):
        return self.darwin.get_camera_image(step=step)
    
    def print_gps(self):
        return self.darwin.get_gps_values()

    def get_touch_sensor_value(self, sensor_name):
        return self.darwin.get_touch_sensor_value(sensor_name)

    def lock_grasp(self):
        """锁定夹爪位置，确保夹爪保持闭合状态"""
        return self.darwin.lock_grasp()
    def is_collision(self):
        """检查是否发生碰撞
        
        通过检测手臂和腿部的触碰传感器来判断碰撞
        检测的传感器包括：
            - arm_L1, arm_R1 (手臂)
            - leg_L1, leg_L2, leg_R1, leg_R2 (腿部)
        
        返回：
            bool: True表示发生碰撞，False表示无碰撞
        """
        return self.darwin.check_collision()  
