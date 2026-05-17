"""Python 3.7 compatible client for the DiffWave GPU service."""

from __future__ import print_function

import os
import pickle
import socket
import struct
import subprocess
import sys
import time

import numpy as np


PICKLE_PROTOCOL = 4
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8876
DEFAULT_MISSION_PYTHON = r"D:\anaconda\envs\mission\python.exe"


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _recv_exact(sock_obj, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock_obj.recv(size - len(data))
        if not chunk:
            raise EOFError("socket closed while receiving packet")
        data.extend(chunk)
    return bytes(data)


def recv_packet(sock_obj):
    header = _recv_exact(sock_obj, 8)
    payload_len = struct.unpack("!Q", header)[0]
    payload = _recv_exact(sock_obj, payload_len)
    return pickle.loads(payload)


def send_packet(sock_obj, obj):
    payload = pickle.dumps(obj, protocol=PICKLE_PROTOCOL)
    header = struct.pack("!Q", len(payload))
    sock_obj.sendall(header + payload)


class DiffWaveGPUClient(object):
    def __init__(
        self,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        mission_python=None,
        project_root=None,
        timeout_s=300.0,
        retry_attempts=120,
        retry_sleep_s=1.0,
        auto_start=True,
    ):
        self.host = str(host)
        self.port = int(port)
        self.mission_python = mission_python or os.environ.get("DIFFWAVE_MISSION_PYTHON", DEFAULT_MISSION_PYTHON)
        self.project_root = project_root or _project_root()
        self.timeout_s = float(timeout_s)
        self.retry_attempts = int(retry_attempts)
        self.retry_sleep_s = float(retry_sleep_s)
        self.auto_start = bool(auto_start)
        self.sock = None
        self.process = None
        self.init_info = None

    def _connect_once(self, timeout_s=2.0):
        sock_obj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_obj.settimeout(float(timeout_s))
        sock_obj.connect((self.host, self.port))
        sock_obj.settimeout(self.timeout_s)
        self.sock = sock_obj

    def start_service(self):
        if self.sock is not None:
            return

        try:
            self._connect_once(timeout_s=1.0)
            print("Using existing DiffWave GPU service at %s:%d" % (self.host, self.port))
            return
        except Exception:
            self.sock = None

        if not self.auto_start:
            return

        if not os.path.isfile(self.mission_python):
            raise RuntimeError("mission Python not found: %s" % self.mission_python)

        env = os.environ.copy()
        old_pythonpath = env.get("PYTHONPATH", "")
        if old_pythonpath:
            env["PYTHONPATH"] = self.project_root + os.pathsep + old_pythonpath
        else:
            env["PYTHONPATH"] = self.project_root

        command = [
            self.mission_python,
            "-u",
            "-m",
            "python_scripts.DiffWave.diffwave_gpu_service",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        print("Starting DiffWave GPU service with mission python")
        print("Mission Python: %s" % self.mission_python)
        print("GPU service endpoint: %s:%d" % (self.host, self.port))
        self.process = subprocess.Popen(command, cwd=self.project_root, env=env)

    def connect(self):
        if self.sock is not None:
            return

        self.start_service()
        if self.sock is not None:
            return

        last_error = None
        for _ in range(self.retry_attempts):
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    "DiffWave GPU service exited early with code %s" % self.process.returncode
                )
            try:
                self._connect_once(timeout_s=self.timeout_s)
                return
            except OSError as exc:
                last_error = exc
                time.sleep(self.retry_sleep_s)

        raise RuntimeError(
            "failed to connect to DiffWave GPU service at %s:%d after %d attempts: %s"
            % (self.host, self.port, self.retry_attempts, last_error)
        )

    def request(self, cmd, **kwargs):
        self.connect()
        send_packet(self.sock, {"cmd": cmd, "kwargs": kwargs})
        reply = recv_packet(self.sock)
        if not reply.get("ok", False):
            raise RuntimeError(
                "DiffWave GPU service error for cmd=%s\n%s\n%s"
                % (cmd, reply.get("error", "unknown error"), reply.get("traceback", ""))
            )
        return reply.get("result")

    def initialize(self, model_path=None, max_steps_per_episode=22):
        result = self.request(
            "init",
            model_path=model_path,
            max_steps_per_episode=int(max_steps_per_episode),
        )
        self.init_info = result
        print("GPU service connected")
        print("Policy device: %s" % result.get("policy_device", "unknown"))
        print("CUDA available: %s" % result.get("cuda_available", "unknown"))
        print("GPU name: %s" % result.get("gpu_name", "unknown"))
        return result

    def runtime_info(self):
        return self.request("runtime_info")

    def choose_catch(
        self,
        image,
        robot_state,
        graph_state=None,
        safety_features=None,
        explore=True,
        explore_noise_std=None,
        q_guidance_probability=1.0,
        action_clip=1.0,
        candidate_count=None,
        deterministic_eval=False,
        deterministic_seed=None,
    ):
        if graph_state is None:
            graph_state = robot_state
        return self.request(
            "choose_catch",
            image=self._pack_array(image, np.float32),
            robot_state=self._pack_array(robot_state, np.float32),
            graph_state=self._pack_array(graph_state, np.float32),
            safety_features=self._pack_optional_array(safety_features, np.float32),
            explore=bool(explore),
            explore_noise_std=explore_noise_std,
            q_guidance_probability=float(q_guidance_probability),
            action_clip=float(action_clip),
            candidate_count=candidate_count,
            deterministic_eval=bool(deterministic_eval),
            deterministic_seed=deterministic_seed,
        )

    def store_catch(
        self,
        state,
        actions,
        reward,
        next_state,
        done,
        values,
        safety_features=None,
        success_flag=False,
        safety_penalty=0.0,
    ):
        return self.request(
            "store_catch",
            state=self._pack_state(state),
            actions=[float(actions[0]), float(actions[1])],
            reward=float(reward),
            next_state=self._pack_state(next_state),
            done=int(done),
            values=[float(values[0]), float(values[1])],
            safety_features=self._pack_optional_array(safety_features, np.float32),
            success_flag=bool(success_flag),
            safety_penalty=float(safety_penalty),
        )

    def learn_catch(self):
        return self.request("learn_catch")

    def choose_tai(
        self,
        image,
        robot_state,
        graph_state=None,
        explore=True,
        explore_noise_std=None,
        q_guidance_probability=1.0,
        action_clip=1.0,
        candidate_count=None,
    ):
        if graph_state is None:
            graph_state = robot_state
        return self.request(
            "choose_tai",
            image=self._pack_array(image, np.float32),
            robot_state=self._pack_array(robot_state, np.float32),
            graph_state=self._pack_array(graph_state, np.float32),
            explore=bool(explore),
            explore_noise_std=explore_noise_std,
            q_guidance_probability=float(q_guidance_probability),
            action_clip=float(action_clip),
            candidate_count=candidate_count,
        )

    def store_tai(
        self,
        state,
        actions,
        reward,
        next_state,
        done,
        values,
        success_flag=False,
        safety_penalty=0.0,
    ):
        return self.request(
            "store_tai",
            state=self._pack_state(state),
            actions=[float(actions[0]), float(actions[1]), float(actions[2])],
            reward=float(reward),
            next_state=self._pack_state(next_state),
            done=int(done),
            values=[float(values[0]), float(values[1]), float(values[2])],
            success_flag=bool(success_flag),
            safety_penalty=float(safety_penalty),
        )

    def learn_tai(self):
        return self.request("learn_tai")

    def save_catch_checkpoint(
        self,
        episode,
        episode_return,
        success_flag,
        test_success_rate,
        successful_test_episodes,
        num_test_episodes,
        save_best=False,
    ):
        return self.request(
            "save_catch_checkpoint",
            episode=int(episode),
            episode_return=float(episode_return),
            success_flag=int(success_flag),
            test_success_rate=float(test_success_rate),
            successful_test_episodes=int(successful_test_episodes),
            num_test_episodes=int(num_test_episodes),
            save_best=bool(save_best),
        )

    def save_tai_checkpoint(self, total_episode, tai_episode):
        return self.request(
            "save_tai_checkpoint",
            total_episode=int(total_episode),
            tai_episode=int(tai_episode),
        )

    def set_mode(self, stage="catch", mode="train"):
        return self.request("set_mode", stage=str(stage), mode=str(mode))

    def close(self):
        result = None
        if self.sock is not None:
            try:
                result = self.request("close")
            except Exception:
                result = None
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

        if self.process is not None:
            try:
                self.process.wait(timeout=5.0)
            except Exception:
                try:
                    self.process.terminate()
                except Exception:
                    pass
            self.process = None
        return result

    def _pack_array(self, value, dtype):
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            try:
                value = value.detach().cpu().numpy()
            except Exception:
                pass
        return np.asarray(value, dtype=dtype).copy()

    def _pack_optional_array(self, value, dtype):
        if value is None:
            return None
        return self._pack_array(value, dtype)

    def _pack_state(self, state):
        if isinstance(state, (list, tuple)) and len(state) >= 3:
            return [
                self._pack_array(state[0], np.float16),
                self._pack_array(state[1], np.float32),
                self._pack_array(state[2], np.float32),
            ]
        return [
            self._pack_array(state, np.float16),
            np.zeros(20, dtype=np.float32),
            np.zeros(19, dtype=np.float32),
        ]

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    client = DiffWaveGPUClient(auto_start=False)
    print(client.runtime_info())


if __name__ == "__main__":
    sys.exit(main())
