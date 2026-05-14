# -*- coding: utf-8 -*-
"""
Webots controller side for the route-3 dual-Python architecture.

Python runtime target
---------------------
- Webots controller process: Python 3.7.12

Design rules
------------
- This file runs *inside* Webots as the controller script.
- It exposes the simulation through the same Environment/Webots_interfaces
  stack used by the PPO/DiffWave route, so RobotRun1/RobotRun2 remain the
  source of stage dynamics, reward, and done logic.
- It exposes a tiny RPC surface over a TCP socket:
    reset(seed=None, options=None)
    step_grasp(action)
    step_tai(action)
    close()

Protocol
--------
- Length-prefixed pickle payloads.
- Pickle protocol is pinned to 4 so it is safe between Python 3.7 and 3.11.
"""
from __future__ import print_function

import argparse
import pickle
import socket
import struct
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from controller import Robot
import sys
sys.path.append('E:\\webotsone\\Multi-Stage_Hybrid_Training')
sys.path.append('C:\\Users\\lenovo\\AppData\\Local\\Programs\\Webots\\projects\\robots\\robotis\\darwin-op\\libraries\\managers')

from python_scripts.Webots_interfaces import Environment

# ---------------------------------------------------------------------------
# Project goals (try project config first, then safe fallbacks)
# ---------------------------------------------------------------------------

try:
    from python_scripts.Project_config import gps_goal as PROJECT_GRASP_GOAL  # type: ignore
    from python_scripts.Project_config import gps_goal1 as PROJECT_TAI_GOAL  # type: ignore
except Exception:
    PROJECT_GRASP_GOAL = (0.2, 0.175)
    PROJECT_TAI_GOAL = (0.27, 0.225)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
PICKLE_PROTOCOL = 4
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

MOTOR_NAMES = (
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL",
    "ArmLowerR", "ArmLowerL", "PelvYR", "PelvYL",
    "PelvR", "PelvL", "LegUpperR", "LegUpperL",
    "LegLowerR", "LegLowerL", "AnkleR", "AnkleL",
    "FootR", "FootL", "Neck", "Head", "GraspL", "GraspR",
)

JOINT_LIMITS = [
    [-3.14, 3.14], [-3.14, 2.85], [-0.68, 2.30], [-2.25, 0.77],
    [-1.65, 1.16], [-1.18, 1.63], [-2.42, 0.66], [-0.69, 2.50],
    [-1.01, 1.01], [-1.00, 0.93], [-1.77, 0.45], [-0.50, 1.68],
    [-0.02, 2.25], [-2.25, 0.03], [-1.24, 1.38], [-1.39, 1.22],
    [-0.68, 1.04], [-1.02, 0.60], [-1.81, 1.81], [-0.36, 0.94],
]

ACC_LOW = [480.0, 450.0, 580.0]
ACC_HIGH = [560.0, 530.0, 700.0]
GYRO_LOW = [500.0, 500.0, 500.0]
GYRO_HIGH = [520.0, 520.0, 520.0]

DEFAULT_GRASP_TRIGGER_STEP = 19
DEFAULT_MAX_GRASP_STEPS = 120
DEFAULT_MAX_TAI_STEPS = 21

IMAGE_SIZE = (128, 128)


# ---------------------------------------------------------------------------
# Socket helpers
# ---------------------------------------------------------------------------
def _recv_exact(sock_obj, size):
    data = b""
    while len(data) < size:
        chunk = sock_obj.recv(size - len(data))
        if not chunk:
            raise EOFError("socket closed while receiving packet")
        data += chunk
    return data


def recv_packet(sock_obj):
    header = _recv_exact(sock_obj, 8)
    payload_len = struct.unpack("!Q", header)[0]
    payload = _recv_exact(sock_obj, payload_len)
    return pickle.loads(payload)



def send_packet(sock_obj, obj):
    payload = pickle.dumps(obj, protocol=PICKLE_PROTOCOL)
    header = struct.pack("!Q", len(payload))
    sock_obj.sendall(header + payload)


# ---------------------------------------------------------------------------
# Legacy standalone adapter kept for reference; Route3WebotsEnv uses Environment.
# ---------------------------------------------------------------------------
class DarwinAdapter(object):
    def __init__(self, robot, rng=None):
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.rng = rng if rng is not None else np.random.RandomState()

        self.motor_names = MOTOR_NAMES
        self.motors = []
        self.motor_sensors = []
        self._init_motors()

        self.camera = robot.getDevice("Camera")
        self.accelerometer = robot.getDevice("Accelerometer")
        self.gyro = robot.getDevice("Gyro")
        self.left_gps1 = robot.getDevice("left_gps1")
        self.right_gps1 = robot.getDevice("right_gps1")
        self.left_gps2 = robot.getDevice("left_gps2")
        self.right_gps2 = robot.getDevice("right_gps2")
        self.foot_gps1 = robot.getDevice("foot_gps1")

        self.touch_sensors = {}
        self._init_touch_sensors()
        self._enable_sensors()

    def _init_motors(self):
        for name in self.motor_names:
            motor = self.robot.getDevice(name)
            sensor = self.robot.getDevice(name + "S")
            sensor.enable(self.timestep)
            self.motors.append(motor)
            self.motor_sensors.append(sensor)

    def _init_touch_sensors(self):
        sensor_map = {
            "grasp_L1": "touch_grasp_L1",
            "grasp_L1_1": "touch_grasp_L1_1",
            "grasp_L1_2": "touch_grasp_L1_2",
            "grasp_R1": "touch_grasp_R1",
            "grasp_R1_1": "touch_grasp_R1_1",
            "grasp_R1_2": "touch_grasp_R1_2",
            "foot_L1": "touch_foot_L1",
            "foot_L2": "touch_foot_L2",
            "foot_L3": "touch_foot_L3",
            "foot_R1": "touch_foot_R1",
            "foot_R2": "touch_foot_R2",
            "arm_L1": "touch_arm_L1",
            "arm_R1": "touch_arm_R1",
            "leg_L1": "touch_leg_L1",
            "leg_L2": "touch_leg_L2",
            "leg_R1": "touch_leg_R1",
            "leg_R2": "touch_leg_R2",
        }
        for key, device_name in sensor_map.items():
            sensor = self.robot.getDevice(device_name)
            sensor.enable(self.timestep)
            self.touch_sensors[key] = sensor

    def _enable_sensors(self):
        self.camera.enable(self.timestep)
        self.accelerometer.enable(self.timestep)
        self.gyro.enable(self.timestep)
        self.left_gps1.enable(self.timestep)
        self.right_gps1.enable(self.timestep)
        self.left_gps2.enable(self.timestep)
        self.right_gps2.enable(self.timestep)
        self.foot_gps1.enable(self.timestep)

    def step(self, n=1):
        for _ in range(int(n)):
            if self.robot.step(self.timestep) == -1:
                return False
        return True

    def wait_ms(self, duration_ms):
        elapsed = 0
        while elapsed < int(duration_ms):
            if self.robot.step(self.timestep) == -1:
                return False
            elapsed += self.timestep
        return True

    def set_all_velocity(self, velocity=1.0):
        for motor in self.motors:
            motor.setVelocity(float(velocity))

    def _set_initial_pose(self, a, b):
        pose_config = {
            "GraspL": 1.0, "GraspR": 1.0,
            "Neck": 0.2, "Head": 0.0,
            "PelvYL": 0.0, "PelvL": 0.0,
            "LegUpperL": 0.5, "LegLowerL": -0.2,
            "AnkleL": 0.0, "FootL": 0.0,
            "PelvYR": 0.0, "PelvR": 0.0,
            "LegUpperR": -0.5, "LegLowerR": 0.2,
            "AnkleR": 0.0, "FootR": 0.0,
            "ShoulderL": a, "ArmUpperL": 0.67, "ArmLowerL": -b,
            "ShoulderR": -a, "ArmUpperR": -0.67, "ArmLowerR": b,
        }
        for name, position in pose_config.items():
            idx = self.motor_names.index(name)
            self.motors[idx].setPosition(float(position))

    def robot_reset(self, seed=None):
        if seed is not None:
            self.rng.seed(int(seed))
        a = -float(self.rng.random_sample()) * 0.4 + 0.7
        b = -float(self.rng.random_sample()) * 0.4 + 1.0
        self._set_initial_pose(a, b)
        self.set_all_velocity(1.0)
        self.step(24)

    def get_robot_state(self):
        return [sensor.getValue() for sensor in self.motor_sensors[:-2]]

    def get_gps_values(self):
        return {
            "left_gps1": np.asarray(self.left_gps1.getValues(), dtype=np.float32),
            "right_gps1": np.asarray(self.right_gps1.getValues(), dtype=np.float32),
            "left_gps2": np.asarray(self.left_gps2.getValues(), dtype=np.float32),
            "right_gps2": np.asarray(self.right_gps2.getValues(), dtype=np.float32),
            "foot_gps1": np.asarray(self.foot_gps1.getValues(), dtype=np.float32),
        }

    def get_touch_values(self):
        out = {}
        for key, sensor in self.touch_sensors.items():
            out[key] = float(sensor.getValue())
        return out

    def get_acc(self):
        return np.asarray(self.accelerometer.getValues(), dtype=np.float32)

    def get_gyro(self):
        return np.asarray(self.gyro.getValues(), dtype=np.float32)

    def check_acceleration_and_gyro(self):
        acc = self.accelerometer.getValues()
        gyro = self.gyro.getValues()
        for i in range(3):
            if not (ACC_LOW[i] < acc[i] < ACC_HIGH[i]):
                return False
            if not (GYRO_LOW[i] < gyro[i] < GYRO_HIGH[i]):
                return False
        return True

    def check_joint_limits(self, positions):
        for idx, pos in enumerate(positions):
            low, high = JOINT_LIMITS[idx]
            if pos < low or pos > high:
                return False
        return True

    def check_collision(self):
        collision_keys = ["arm_L1", "arm_R1", "leg_L1", "leg_L2", "leg_R1", "leg_R2"]
        return any(self.touch_sensors[key].getValue() for key in collision_keys)

    def grasp_contact_flags(self):
        left_keys = ["grasp_L1", "grasp_L1_1", "grasp_L1_2"]
        right_keys = ["grasp_R1", "grasp_R1_1", "grasp_R1_2"]
        left_any = any(self.touch_sensors[key].getValue() for key in left_keys)
        right_any = any(self.touch_sensors[key].getValue() for key in right_keys)
        return bool(left_any), bool(right_any)

    def close_gripper(self, wait_ms=2000):
        self.motors[20].setPosition(-0.5)
        self.motors[21].setPosition(-0.5)
        self.wait_ms(wait_ms)

    def open_gripper(self, wait_ms=320):
        self.motors[20].setPosition(1.0)
        self.motors[21].setPosition(1.0)
        self.wait_ms(wait_ms)

    def execute_timed_motion(self, motor_positions, duration_ms, velocity=1.0):
        for name, position in motor_positions.items():
            idx = self.motor_names.index(name)
            self.motors[idx].setPosition(float(position))
            self.motors[idx].setVelocity(float(velocity))
        self.wait_ms(duration_ms)

    def tai_leg_L1(self):
        self.execute_timed_motion({
            "LegLowerL": -0.7,
            "AnkleL": -0.5,
        }, 2000, 1.0)

    def tai_leg_L2(self):
        self.execute_timed_motion({
            "LegUpperL": 1.65,
            "LegLowerL": -2.2,
            "AnkleL": -0.85,
        }, 2000, 2.0)

    def set_left_leg_initpose(self):
        self.execute_timed_motion({
            "LegUpperL": 1.65,
            "LegLowerL": -2.2,
            "AnkleL": -0.85,
        }, 1500, 1.5)
        self.execute_timed_motion({
            "LegUpperL": 0.4,
            "LegLowerL": -0.7,
            "AnkleL": -0.7,
        }, 1500, 1.0)
        self.execute_timed_motion({
            "LegLowerL": -0.1,
            "AnkleL": -0.15,
        }, 1500, 1.0)

    def capture_image_gray_128(self):
        width = int(self.camera.getWidth())
        height = int(self.camera.getHeight())
        raw = self.camera.getImage()
        if raw is None:
            return np.zeros(IMAGE_SIZE, dtype=np.float32)

        bgra = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
        rgb = bgra[:, :, 2::-1]
        gray = (
            0.299 * rgb[:, :, 0].astype(np.float32)
            + 0.587 * rgb[:, :, 1].astype(np.float32)
            + 0.114 * rgb[:, :, 2].astype(np.float32)
        ).astype(np.uint8)

        if (width, height) != IMAGE_SIZE:
            gray = np.array(Image.fromarray(gray).resize(IMAGE_SIZE, Image.BILINEAR), dtype=np.uint8)
        return gray.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Route-3 environment logic
# ---------------------------------------------------------------------------
class Route3WebotsEnv(object):
    def __init__(self):
        self.env = Environment()
        self.robot = self.env.robot
        self.timestep = int(self.robot.getBasicTimeStep())
        self.rng = np.random.RandomState()
        self.darwin = self.env.darwin

        self.grasp_goal = np.asarray(PROJECT_GRASP_GOAL, dtype=np.float32)
        self.tai_goal = np.asarray(PROJECT_TAI_GOAL, dtype=np.float32)
        self.grasp_trigger_step = int(DEFAULT_GRASP_TRIGGER_STEP)
        self.max_grasp_steps = int(DEFAULT_MAX_GRASP_STEPS)
        self.max_tai_steps = int(DEFAULT_MAX_TAI_STEPS)

        self.closed = False
        self.episode_index = 0
        self._reset_runtime_state()

    def _reset_runtime_state(self):
        self.current_stage = "idle"
        self.grasp_step_count = 0
        self.tai_step_count = 0
        self.prev_grasp_distance = None
        self.grasp_success = False
        self.tai_prepared = False
        self.last_seed = None
        self._grasp_catch_flag = 0.0

    def _apply_reset_options(self, options):
        options = options or {}
        if "grasp_goal" in options:
            self.grasp_goal = np.asarray(options["grasp_goal"], dtype=np.float32)
        if "tai_goal" in options:
            self.tai_goal = np.asarray(options["tai_goal"], dtype=np.float32)
        if "grasp_trigger_step" in options:
            self.grasp_trigger_step = int(options["grasp_trigger_step"])
        if "max_grasp_steps" in options:
            self.max_grasp_steps = int(options["max_grasp_steps"])
        if "max_tai_steps" in options:
            self.max_tai_steps = int(options["max_tai_steps"])

    def _wait_until_stable(self, max_retry=60):
        return bool(self.env.wait_until_stable(max_retry=int(max_retry), verbose=False))

    def _recover_if_unstable(self, max_retry=20):
        if self.darwin.check_acceleration_and_gyro():
            return True
        return self._wait_until_stable(max_retry=max_retry)

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _safe_int(value, default=0):
        try:
            return int(value)
        except Exception:
            return int(default)

    def _gps_tuple(self):
        try:
            raw_gps = self.env.print_gps()
        except Exception:
            raw_gps = ()

        gps_values = []
        for item in raw_gps:
            arr = np.asarray(item, dtype=np.float32).reshape(-1)
            if arr.size < 3:
                padded = np.zeros(3, dtype=np.float32)
                padded[: arr.size] = arr
                arr = padded
            gps_values.append(arr[:3].copy())

        while len(gps_values) < 5:
            gps_values.append(np.zeros(3, dtype=np.float32))
        return tuple(gps_values[:5])

    def _gps_dict(self):
        gps0, gps1, gps2, gps3, gps4 = self._gps_tuple()
        return {
            "left_gps1": gps0,
            "right_gps1": gps1,
            "left_gps2": gps2,
            "right_gps2": gps3,
            "foot_gps1": gps4,
        }

    def _touch_values(self):
        try:
            touch = self.darwin.get_touch_values()
        except Exception:
            touch = {}
        out = {}
        for key, value in touch.items():
            out[str(key)] = self._safe_float(value)
        return out

    def _grasp_contacts(self):
        try:
            contact = self.darwin.check_grasp_contact()
            return bool(contact.get("left", False)), bool(contact.get("right", False))
        except Exception:
            touch = self._touch_values()
            left = bool(
                touch.get("grasp_L1", 0.0)
                or touch.get("grasp_L1_1", 0.0)
                or touch.get("grasp_L1_2", 0.0)
            )
            right = bool(
                touch.get("grasp_R1", 0.0)
                or touch.get("grasp_R1_1", 0.0)
                or touch.get("grasp_R1_2", 0.0)
            )
            return left, right

    def _grasp_distance(self):
        left_gps1 = self._gps_tuple()[0]
        if left_gps1.shape[0] < 3:
            return 1.0
        dy = float(self.grasp_goal[0] - left_gps1[1])
        dz = float(self.grasp_goal[1] - left_gps1[2])
        return float(np.sqrt(dy * dy + dz * dz))

    def _tai_distance(self):
        foot = self._gps_tuple()[4]
        if foot.shape[0] < 3:
            return 1.0
        dx = float(self.tai_goal[0] - foot[1])
        dy = float(self.tai_goal[1] - foot[2])
        return float(np.sqrt(dx * dx + dy * dy))

    def _build_obs(self):
        try:
            image, _ = self.env.get_img(
                self.grasp_step_count if self.current_stage != "tai" else self.tai_step_count,
                [],
            )
            image = np.asarray(image, dtype=np.float32)
        except Exception:
            image = np.zeros(IMAGE_SIZE, dtype=np.float32)

        try:
            state = np.asarray(self.env.get_robot_state(), dtype=np.float32).reshape(-1)
        except Exception:
            state = np.zeros(20, dtype=np.float32)

        gps = self._gps_dict()
        touch = self._touch_values()
        obs = {
            "image": image.copy(),
            "robot_state": state.copy(),
            "graph_state": state[:19].copy(),
            "gps": gps,
            "touch": touch,
            "acc": np.asarray(self.darwin.accelerometer.getValues(), dtype=np.float32),
            "gyro": np.asarray(self.darwin.gyro.getValues(), dtype=np.float32),
            "stage": self.current_stage,
            "grasp_step": int(self.grasp_step_count),
            "tai_step": int(self.tai_step_count),
        }
        return obs

    def reset(self, seed=None, options=None):
        self._apply_reset_options(options)
        self._reset_runtime_state()
        self.episode_index += 1
        self.last_seed = None if seed is None else int(seed)
        if self.last_seed is not None:
            self.rng.seed(self.last_seed)
            np.random.seed(self.last_seed)

        self.env.reset()
        self.darwin = self.env.darwin

        self.current_stage = "grasp"
        obs = self._build_obs()
        info = {
            "episode": int(self.episode_index),
            "seed": self.last_seed,
            "grasp_goal": self.grasp_goal.copy(),
            "tai_goal": self.tai_goal.copy(),
            "grasp_trigger_step": int(self.grasp_trigger_step),
            "max_grasp_steps": int(self.max_grasp_steps),
            "max_tai_steps": int(self.max_tai_steps),
        }
        return obs, info

    def _normalize_grasp_result(self, result):
        if not isinstance(result, (tuple, list)):
            raise RuntimeError("Environment.step returned an unsupported result: %r" % (result,))
        if len(result) >= 6:
            next_state, reward, done, good, goal, count = result[:6]
            return next_state, reward, done, good, goal, count
        if len(result) >= 4:
            next_state, reward, done, catch_success = result[:4]
            good = 1
            goal = 1 if catch_success else 0
            count = 0 if catch_success else 1
            return next_state, reward, done, good, goal, count
        raise RuntimeError("Environment.step returned too few values: %r" % (result,))

    def _normalize_tai_result(self, result):
        if not isinstance(result, (tuple, list)) or len(result) < 6:
            raise RuntimeError("Environment.step2 returned an unsupported result: %r" % (result,))
        return result[:6]

    def step_grasp(self, action):
        if self.closed:
            raise RuntimeError("environment already closed")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 2:
            raise ValueError("step_grasp expects action_dim=2")

        self.current_stage = "grasp"
        stage_step = int(self.grasp_step_count)

        if not self._recover_if_unstable(max_retry=20):
            reward = 0.0
            terminated = True
            truncated = False
            good = 0
            goal = 0
            count = 0
            reason = "unstable_before_step"
        else:
            gps_values = self._gps_tuple()
            result = self.env.step(
                self.env.get_robot_state(),
                float(action[0]),
                float(action[1]),
                stage_step,
                self._grasp_catch_flag,
                gps_values[0],
                gps_values[1],
                gps_values[2],
                gps_values[3],
                "dp_grasp_%d_%d.png" % (int(self.episode_index), int(stage_step)),
            )
            _next_state, reward, done, good, goal, count = self._normalize_grasp_result(result)
            reward = self._safe_float(reward)
            terminated = bool(done)
            truncated = False
            good = self._safe_int(good, 1)
            goal = self._safe_int(goal, 0)
            count = self._safe_int(count, 1)
            success = bool(goal == 1)
            self.grasp_success = bool(success)
            self._grasp_catch_flag = 1.0 if terminated else self._grasp_catch_flag

            if success:
                reason = "grasp_success"
                self.current_stage = "grasp_done"
            elif terminated and stage_step <= 2 and reward <= 0.0:
                reason = "invalid_abort"
                self.current_stage = "grasp_done"
            elif terminated and good == 0:
                reason = "invalid_state"
                self.current_stage = "grasp_done"
            elif terminated:
                reason = "grasp_done"
                self.current_stage = "grasp_done"
            else:
                reason = "running"

        self.grasp_step_count += 1
        if (self.grasp_step_count >= self.max_grasp_steps) and (not terminated):
            truncated = True
            reason = "max_grasp_steps"
            self.current_stage = "grasp_done"

        obs = self._build_obs()
        left_contact, right_contact = self._grasp_contacts()
        distance = self._grasp_distance()
        self.prev_grasp_distance = distance
        info = {
            "stage": "grasp",
            "step": int(self.grasp_step_count),
            "reason": reason,
            "good": int(good),
            "goal": int(self.grasp_success),
            "count": 0 if self.grasp_success else 1,
            "success": bool(self.grasp_success),
            "distance": float(distance),
            "left_contact": bool(left_contact),
            "right_contact": bool(right_contact),
            "grasp_trigger_step": int(self.grasp_trigger_step),
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def step_tai(self, action):
        if self.closed:
            raise RuntimeError("environment already closed")
        if not self.grasp_success:
            raise RuntimeError("step_tai can only be called after successful grasp")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 3:
            raise ValueError("step_tai expects action_dim=3")

        if not self.tai_prepared:
            self.env.darwin.tai_leg_L1()
            self.env.darwin.tai_leg_L2()
            self.tai_prepared = True

        self.current_stage = "tai"

        if not self._recover_if_unstable(max_retry=20):
            reward = 0.0
            terminated = True
            truncated = False
            good = 0
            goal = 0
            count = 0
            success = False
            reason = "unstable_before_step"
        else:
            gps_values = self._gps_tuple()
            result = self.env.step2(
                self.env.get_robot_state(),
                float(action[0]),
                float(action[1]),
                float(action[2]),
                int(self.tai_step_count),
                1.0,
                gps_values[4],
                gps_values[0],
                gps_values[1],
                gps_values[2],
                gps_values[3],
            )
            _next_state, reward, done, good, goal, count = self._normalize_tai_result(result)
            reward = self._safe_float(reward)
            terminated = bool(done)
            truncated = False
            good = self._safe_int(good, 1)
            goal = self._safe_int(goal, 0)
            count = self._safe_int(count, 1)
            success = bool(goal == 1)

            if success:
                reason = "tai_success"
            elif terminated and good == 0:
                reason = "invalid_state"
            elif terminated and count == 0:
                reason = "tai_constraint"
            elif terminated:
                reason = "tai_done"
            else:
                reason = "running"

        self.tai_step_count += 1
        if (self.tai_step_count >= self.max_tai_steps) and (not terminated):
            truncated = True
            reason = "max_tai_steps"
            self.current_stage = "tai_done"
        elif terminated:
            self.current_stage = "tai_done"

        obs = self._build_obs()
        info = {
            "stage": "tai",
            "step": int(self.tai_step_count),
            "reason": reason,
            "good": int(good),
            "goal": int(goal),
            "count": int(count),
            "success": bool(success),
            "distance": float(self._tai_distance()),
            "prepared": bool(self.tai_prepared),
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.closed = True
        self.current_stage = "closed"
        return {"closed": True, "episode": int(self.episode_index)}


# ---------------------------------------------------------------------------
# RPC server
# ---------------------------------------------------------------------------
class WebotsRPCServer(object):
    def __init__(self, host, port):
        self.host = str(host)
        self.port = int(port)
        self.env = Route3WebotsEnv()
        self._stop = False

    def _dispatch(self, request):
        cmd = request.get("cmd")
        kwargs = request.get("kwargs", {})

        if cmd == "reset":
            obs, info = self.env.reset(seed=kwargs.get("seed"), options=kwargs.get("options"))
            return {"obs": obs, "info": info}
        elif cmd == "step_grasp":
            obs, reward, terminated, truncated, info = self.env.step_grasp(kwargs.get("action"))
            return {
                "obs": obs,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
            }
        elif cmd == "step_tai":
            obs, reward, terminated, truncated, info = self.env.step_tai(kwargs.get("action"))
            return {
                "obs": obs,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
            }
        elif cmd == "close":
            self._stop = True
            return self.env.close()
        else:
            raise ValueError("unknown cmd: %s" % (cmd,))

    def serve_forever(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(1)
        print("[WebotsEnvServer] listening on %s:%d" % (self.host, self.port))

        try:
            while not self._stop:
                print("[WebotsEnvServer] waiting for external trainer connection...")
                conn, addr = server_sock.accept()
                print("[WebotsEnvServer] connected by %s:%s" % (addr[0], addr[1]))
                try:
                    while not self._stop:
                        request = recv_packet(conn)
                        try:
                            result = self._dispatch(request)
                            send_packet(conn, {"ok": True, "result": result})
                        except Exception as exc:
                            send_packet(conn, {
                                "ok": False,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            })
                            if request.get("cmd") == "close":
                                self._stop = True
                except EOFError:
                    print("[WebotsEnvServer] client disconnected")
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
        finally:
            try:
                server_sock.close()
            except Exception:
                pass
            print("[WebotsEnvServer] shutdown complete")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Webots controller env server (route 3 dual-Python)")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    server = WebotsRPCServer(host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
