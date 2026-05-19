"""WaveGrad GPU algorithm service.

This process is launched by the Webots controller, but it runs with the
mission Python environment. Webots stays in charge of the simulation loop;
this service only owns WaveGrad policies, replay buffers, learning, and
checkpoints.
"""

from __future__ import annotations

import argparse
import glob
import heapq
import os
import pickle
import re
import socket
import struct
import sys
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from python_scripts.WaveGrad.WaveGrad_policy import WaveGradAgent
from python_scripts.WaveGrad.WaveGrad_policy_2 import WaveGradTaiAgent
from python_scripts.Project_config import device, path_list


PICKLE_PROTOCOL = 4
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8877


def _recv_exact(sock_obj: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock_obj.recv(size - len(data))
        if not chunk:
            raise EOFError("socket closed while receiving packet")
        data.extend(chunk)
    return bytes(data)


def recv_packet(sock_obj: socket.socket) -> Dict[str, Any]:
    header = _recv_exact(sock_obj, 8)
    payload_len = struct.unpack("!Q", header)[0]
    payload = _recv_exact(sock_obj, payload_len)
    return pickle.loads(payload)


def send_packet(sock_obj: socket.socket, obj: Dict[str, Any]) -> None:
    payload = pickle.dumps(obj, protocol=PICKLE_PROTOCOL)
    header = struct.pack("!Q", len(payload))
    sock_obj.sendall(header + payload)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _extract_first_number(file_path: str) -> int:
    numbers = re.findall(r"\d+", os.path.basename(file_path))
    return int(numbers[0]) if numbers else 0


def _catch_checkpoint_sort_key(file_path: str) -> Tuple[int, int]:
    name = os.path.basename(file_path)
    success_match = re.search(r"wavegrad_model_success_(\d+)_(\d+)_", name)
    if success_match:
        return int(success_match.group(1)), int(success_match.group(2))
    model_match = re.search(r"wavegrad_model_(\d+)\.ckpt", name)
    if model_match:
        return int(model_match.group(1)), 0
    return _extract_first_number(file_path), 0


def _parse_catch_success_checkpoint_name(file_path: str) -> Tuple[int, int, Optional[float]]:
    name = os.path.basename(file_path)
    match = re.search(
        r"wavegrad_model_success_(\d+)_(\d+)_([+-]?\d+(?:p\d+)?|[+-]?\d+(?:\.\d+)?)\.ckpt$",
        name,
    )
    if not match:
        return _catch_checkpoint_sort_key(file_path)[0], 0, None
    score_text = match.group(3).replace("p", ".")
    try:
        score = float(score_text)
    except ValueError:
        score = None
    return int(match.group(1)), int(match.group(2)), score


def _checkpoint_float(checkpoint: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key not in checkpoint:
            continue
        try:
            return float(checkpoint[key])
        except (TypeError, ValueError):
            continue
    return None


def _checkpoint_int(checkpoint: Dict[str, Any], key: str) -> Optional[int]:
    if key not in checkpoint:
        return None
    try:
        return int(checkpoint[key])
    except (TypeError, ValueError):
        return None


def _torch_load_checkpoint(file_path: str, map_location: Any) -> Any:
    try:
        return torch.load(file_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(file_path, map_location=map_location)


def _read_catch_checkpoint_rank(file_path: str) -> Tuple[Optional[float], int]:
    filename_episode, _, filename_score = _parse_catch_success_checkpoint_name(file_path)
    if filename_score is not None:
        return filename_score, filename_episode

    episode = filename_episode
    try:
        checkpoint = _torch_load_checkpoint(file_path, map_location="cpu")
    except Exception as exc:
        print("Could not inspect WaveGrad catch checkpoint %s: %s" % (file_path, exc), flush=True)
        return None, episode

    if not isinstance(checkpoint, dict):
        return None, episode

    checkpoint_episode = _checkpoint_int(checkpoint, "episode")
    if checkpoint_episode is not None:
        episode = checkpoint_episode

    score = _checkpoint_float(checkpoint, ("rank_score", "test_success_rate"))
    if score is None and os.path.basename(file_path) == "best_catch_WaveGrad.pth":
        score = _checkpoint_float(checkpoint, ("best_test_success_rate",))
    return score, episode


def _select_best_catch_checkpoint(base_dir: str) -> Optional[Tuple[str, Optional[float], int]]:
    model_files = set(glob.glob(os.path.join(base_dir, "wavegrad_model_*.ckpt")))
    best_path = os.path.join(base_dir, "best_catch_WaveGrad.pth")
    if os.path.isfile(best_path):
        model_files.add(best_path)
    if not model_files:
        return None

    best_candidate: Optional[Tuple[Tuple[float, int, float], str, Optional[float], int]] = None
    for model_file in model_files:
        score, episode = _read_catch_checkpoint_rank(model_file)
        rank_score = score if score is not None else -1.0
        try:
            mtime = os.path.getmtime(model_file)
        except OSError:
            mtime = 0.0
        rank_key = (rank_score, episode, mtime)
        if best_candidate is None or rank_key > best_candidate[0]:
            best_candidate = (rank_key, model_file, score, episode)

    if best_candidate is None:
        return None
    _, model_file, score, episode = best_candidate
    return model_file, score, episode


def _parse_tai_episode(file_path: str) -> Tuple[int, int]:
    match = re.search(r"wavegrad_model_tai_(\d+)_(\d+)\.ckpt", os.path.basename(file_path))
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, _extract_first_number(file_path)


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer) -> None:
    target_device = torch.device(device)
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(target_device)


def _load_policy_state(
    agent: Any,
    checkpoint: Dict[str, Any],
    policy_keys: Iterable[str],
    optimizer_keys: Iterable[str],
    critic_keys: Iterable[str] = (),
    critic_optimizer_keys: Iterable[str] = (),
) -> None:
    for key in policy_keys:
        if key in checkpoint:
            _load_compatible_module_state(agent.policy, checkpoint[key], key)
            break

    for key in optimizer_keys:
        if key in checkpoint:
            try:
                agent.optimizer.load_state_dict(checkpoint[key])
                _move_optimizer_state_to_device(agent.optimizer)
                if hasattr(agent, "_clamp_policy_lr"):
                    agent._clamp_policy_lr()
            except Exception as exc:
                print("Skipped incompatible optimizer state %s: %s" % (key, exc), flush=True)
            break

    if hasattr(agent, "critic"):
        for key in critic_keys:
            if key in checkpoint:
                try:
                    _load_compatible_module_state(agent.critic, checkpoint[key], key)
                except Exception as exc:
                    print("Skipped incompatible critic state %s: %s" % (key, exc), flush=True)
                break

    if hasattr(agent, "critic_optimizer"):
        for key in critic_optimizer_keys:
            if key in checkpoint:
                try:
                    agent.critic_optimizer.load_state_dict(checkpoint[key])
                    _move_optimizer_state_to_device(agent.critic_optimizer)
                except Exception as exc:
                    print("Skipped incompatible critic optimizer state %s: %s" % (key, exc), flush=True)
                break


def _load_compatible_module_state(module: torch.nn.Module, state_dict: Any, label: str) -> None:
    if not isinstance(state_dict, dict):
        module.load_state_dict(state_dict, strict=False)
        return

    current_state = module.state_dict()
    compatible_state = {}
    skipped_keys = []
    for key, value in state_dict.items():
        if key in current_state and hasattr(value, "shape") and current_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped_keys.append(key)

    missing, unexpected = module.load_state_dict(compatible_state, strict=False)
    if skipped_keys:
        print(
            "Loaded compatible %s weights; skipped %d incompatible tensors." % (label, len(skipped_keys)),
            flush=True,
        )
    if unexpected:
        print("Unexpected %s tensors ignored: %d" % (label, len(unexpected)), flush=True)


class ModelRanking:
    def __init__(self, top_n: int = 5):
        self.top_n = int(top_n)
        self.rankings: List[Tuple[float, str]] = []

    def add_and_manage(
        self,
        new_score: float,
        new_checkpoint: Dict[str, Any],
        episode_id: int,
        base_dir: str,
        success_count: int,
    ) -> Optional[str]:
        new_score = float(new_score)
        safe_score = ("%.2f" % new_score).replace(".", "p")
        filename = "wavegrad_model_success_%d_%d_%s.ckpt" % (episode_id, success_count, safe_score)
        final_save_path = os.path.join(base_dir, filename)
        should_save = len(self.rankings) < self.top_n or new_score > self.rankings[0][0]
        if not should_save:
            print("Model %d score %.2f%% did not enter top %d." % (episode_id, new_score, self.top_n), flush=True)
            return None

        if len(self.rankings) >= self.top_n:
            worst_score, worst_path = heapq.heappop(self.rankings)
            try:
                os.remove(worst_path)
                print("Removed old ranked model %s (%.2f%%)." % (worst_path, worst_score), flush=True)
            except FileNotFoundError:
                pass

        torch.save(new_checkpoint, final_save_path)
        heapq.heappush(self.rankings, (new_score, final_save_path))
        print("Saved ranked WaveGrad model: %s" % final_save_path, flush=True)
        return final_save_path


class WaveGradGPUService:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = str(host)
        self.port = int(port)
        self.running = True
        self.max_steps_per_episode = 22
        self.wavegrad_catch: Optional[WaveGradAgent] = None
        self.wavegrad_tai_leg_upper: Optional[WaveGradTaiAgent] = None
        self.wavegrad_tai_leg_lower: Optional[WaveGradTaiAgent] = None
        self.wavegrad_tai_ankle: Optional[WaveGradTaiAgent] = None
        self.model_ranking = ModelRanking(top_n=5)
        self.best_catch_test_success_rate = -1.0

    def runtime_info(self) -> Dict[str, Any]:
        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "none"
        policy_device = "uninitialized"
        if self.wavegrad_catch is not None:
            policy_device = str(next(self.wavegrad_catch.policy.parameters()).device)
        return {
            "sys_executable": sys.executable,
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "project_config_device": device,
            "policy_device": policy_device,
        }

    def init_agents(self, model_path: Optional[str] = None, max_steps_per_episode: int = 22) -> Dict[str, Any]:
        self.max_steps_per_episode = min(int(max_steps_per_episode), 22)
        for folder in (
            path_list["model_path_catch_WaveGrad"],
            path_list["model_path_tai_WaveGrad"],
            path_list["catch_log_path_WaveGrad"],
            path_list["tai_log_path_WaveGrad"],
        ):
            _ensure_dir(folder)

        self.wavegrad_catch = WaveGradAgent(
            node_num=19,
            env_information=None,
            trajectory_len=self.max_steps_per_episode,
            action_dim=2,
            replay_action_clip=0.85,
        )
        self.wavegrad_tai_leg_upper = WaveGradTaiAgent(node_num=19, env_information=None)
        self.wavegrad_tai_leg_lower = WaveGradTaiAgent(node_num=19, env_information=None)
        self.wavegrad_tai_ankle = WaveGradTaiAgent(node_num=19, env_information=None)

        episode_start = self._load_catch_model(model_path)
        tai_episode = self._load_tai_model(default_episode=1)
        info = self.runtime_info()
        info.update(
            {
                "episode_start": int(episode_start),
                "tai_episode": int(tai_episode),
                "best_catch_test_success_rate": float(self.best_catch_test_success_rate),
            }
        )
        print("WaveGrad GPU service initialized.", flush=True)
        print("sys.executable: %s" % info["sys_executable"], flush=True)
        print("torch: %s" % info["torch_version"], flush=True)
        print("torch.cuda.is_available: %s" % info["cuda_available"], flush=True)
        print("Project_config.device: %s" % info["project_config_device"], flush=True)
        print("Policy device: %s" % info["policy_device"], flush=True)
        print("GPU: %s" % info["gpu_name"], flush=True)
        return info

    def _require_agents(self) -> None:
        if self.wavegrad_catch is None:
            self.init_agents(max_steps_per_episode=self.max_steps_per_episode)

    def _load_catch_model(self, model_path: Optional[str]) -> int:
        assert self.wavegrad_catch is not None

        target_model = model_path
        is_specified = bool(model_path)
        selected_score: Optional[float] = None
        selected_episode: Optional[int] = None
        if not target_model:
            selected = _select_best_catch_checkpoint(path_list["model_path_catch_WaveGrad"])
            if selected is None:
                print("No saved WaveGrad catch model found; starting from scratch.", flush=True)
                return 0
            target_model, selected_score, selected_episode = selected

        if not os.path.isfile(target_model):
            print("Specified catch model does not exist: %s" % target_model, flush=True)
            return 0

        try:
            checkpoint = _torch_load_checkpoint(target_model, map_location=device)
            if isinstance(checkpoint, dict):
                if "wavegrad_catch" in checkpoint or "policy_catch" in checkpoint:
                    _load_policy_state(
                        self.wavegrad_catch,
                        checkpoint,
                        policy_keys=("wavegrad_catch", "policy_catch"),
                        optimizer_keys=("optimizer_wavegrad_catch", "optimizer_catch"),
                        critic_keys=("critic_wavegrad_catch",),
                        critic_optimizer_keys=("optimizer_critic_wavegrad_catch",),
                    )
                    self.wavegrad_catch.load_replay_state(checkpoint.get("replay_wavegrad_catch"))
                else:
                    # 仅兼容旧检查点；当前抓取阶段运行时使用wavegrad_catch。
                    legacy_catch_policy_keys = (
                        "wavegrad_shoulder",
                        "policy_shoulder",
                        "wavegrad_arm",
                        "policy_arm",
                    )
                    _load_policy_state(
                        self.wavegrad_catch,
                        checkpoint,
                        policy_keys=legacy_catch_policy_keys,
                        optimizer_keys=(),
                        critic_keys=(),
                        critic_optimizer_keys=(),
                    )
                    print(
                        "Loaded compatible legacy catch policy into joint-action agent; "
                        "critic, optimizer, and replay were reinitialized.",
                        flush=True,
                    )
                checkpoint_episode = int(checkpoint.get("episode", _catch_checkpoint_sort_key(target_model)[0]))
                episode_start = checkpoint_episode + 1 if checkpoint_episode >= 0 else 0
                try:
                    loaded_best = float(
                        checkpoint.get(
                            "best_test_success_rate",
                            checkpoint.get("test_success_rate", self.best_catch_test_success_rate),
                        )
                    )
                    self.best_catch_test_success_rate = max(self.best_catch_test_success_rate, loaded_best)
                except (TypeError, ValueError):
                    pass
                if selected_score is None:
                    selected_score = _checkpoint_float(checkpoint, ("rank_score", "test_success_rate"))
                    if selected_score is None and os.path.basename(target_model) == "best_catch_WaveGrad.pth":
                        selected_score = _checkpoint_float(checkpoint, ("best_test_success_rate",))
            else:
                _load_compatible_module_state(self.wavegrad_catch.policy, checkpoint, "legacy_catch_policy")
                checkpoint_episode = _catch_checkpoint_sort_key(target_model)[0]
                episode_start = checkpoint_episode + 1 if checkpoint_episode >= 0 else 0
            source = "specified" if is_specified else "best-success"
            score_text = "unknown" if selected_score is None else "%.2f%%" % selected_score
            if selected_episode is None:
                selected_episode = checkpoint_episode
            print(
                "Loaded %s WaveGrad catch model: %s; checkpoint episode %d; "
                "test success %s; resume episode %d"
                % (source, target_model, selected_episode, score_text, episode_start),
                flush=True,
            )
            return episode_start
        except Exception as exc:
            print("Failed to load catch model %s: %s" % (target_model, exc), flush=True)
            return 0

    def _load_tai_model(self, default_episode: int = 1) -> int:
        assert self.wavegrad_tai_leg_upper is not None
        assert self.wavegrad_tai_leg_lower is not None
        assert self.wavegrad_tai_ankle is not None

        model_files = glob.glob(os.path.join(path_list["model_path_tai_WaveGrad"], "wavegrad_model_tai_*.ckpt"))
        if not model_files:
            print("No saved WaveGrad tai model found; starting tai training from scratch.", flush=True)
            return int(default_episode)

        latest_model = max(model_files, key=_parse_tai_episode)
        _, ep = _parse_tai_episode(latest_model)
        try:
            checkpoint = _torch_load_checkpoint(latest_model, map_location=device)
            if isinstance(checkpoint, dict):
                _load_policy_state(
                    self.wavegrad_tai_leg_upper,
                    checkpoint,
                    policy_keys=("wavegrad_tai_leg_upper", "policy_LegUpper"),
                    optimizer_keys=("optimizer_wavegrad_tai_leg_upper", "optimizer_LegUpper"),
                    critic_keys=("critic_wavegrad_tai_leg_upper",),
                    critic_optimizer_keys=("optimizer_critic_wavegrad_tai_leg_upper",),
                )
                _load_policy_state(
                    self.wavegrad_tai_leg_lower,
                    checkpoint,
                    policy_keys=("wavegrad_tai_leg_lower", "policy_LegLower"),
                    optimizer_keys=("optimizer_wavegrad_tai_leg_lower", "optimizer_LegLower"),
                    critic_keys=("critic_wavegrad_tai_leg_lower",),
                    critic_optimizer_keys=("optimizer_critic_wavegrad_tai_leg_lower",),
                )
                _load_policy_state(
                    self.wavegrad_tai_ankle,
                    checkpoint,
                    policy_keys=("wavegrad_tai_ankle", "policy_Ankle"),
                    optimizer_keys=("optimizer_wavegrad_tai_ankle", "optimizer_Ankle"),
                    critic_keys=("critic_wavegrad_tai_ankle",),
                    critic_optimizer_keys=("optimizer_critic_wavegrad_tai_ankle",),
                )
                self.wavegrad_tai_leg_upper.load_replay_state(checkpoint.get("replay_wavegrad_tai_leg_upper"))
                self.wavegrad_tai_leg_lower.load_replay_state(checkpoint.get("replay_wavegrad_tai_leg_lower"))
                self.wavegrad_tai_ankle.load_replay_state(checkpoint.get("replay_wavegrad_tai_ankle"))
                tai_episode = int(checkpoint.get("episode", ep))
            else:
                self.wavegrad_tai_leg_upper.policy.load_state_dict(checkpoint, strict=False)
                self.wavegrad_tai_leg_lower.policy.load_state_dict(checkpoint, strict=False)
                self.wavegrad_tai_ankle.policy.load_state_dict(checkpoint, strict=False)
                tai_episode = ep
            print("Loaded WaveGrad tai model: %s; resume tai episode %d" % (latest_model, tai_episode), flush=True)
            return max(int(default_episode), int(tai_episode))
        except Exception as exc:
            print("Failed to load tai model %s: %s" % (latest_model, exc), flush=True)
            return int(default_episode)

    def _as_float_array(self, value: Any, dtype: Any = np.float32) -> np.ndarray:
        if value is None:
            return np.asarray([], dtype=dtype)
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype).copy()

    def _state_triplet(self, state: Any) -> List[np.ndarray]:
        if isinstance(state, (list, tuple)) and len(state) >= 3:
            return [
                self._as_float_array(state[0], np.float16),
                self._as_float_array(state[1], np.float32),
                self._as_float_array(state[2], np.float32),
            ]
        return [
            self._as_float_array(state, np.float16),
            np.zeros(20, dtype=np.float32),
            np.zeros(19, dtype=np.float32),
        ]

    def _action_info(self, agent: Any) -> Dict[str, Any]:
        info = getattr(agent, "last_action_info", {}) or {}
        return {
            "q_guided_used": bool(info.get("q_guided_used", False)),
            "q_guided_action_delta": float(info.get("q_guided_action_delta", 0.0)),
            "critic_updates": int(info.get("critic_updates", 0)),
        }

    def choose_catch(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        self._require_agents()
        assert self.wavegrad_catch is not None

        image = self._as_float_array(kwargs.get("image"), np.float32)
        robot_state = self._as_float_array(kwargs.get("robot_state"), np.float32)
        graph_state = self._as_float_array(kwargs.get("graph_state", robot_state), np.float32)
        safety_features = kwargs.get("safety_features")
        if safety_features is not None:
            safety_features = self._as_float_array(safety_features, np.float32)
        explore = bool(kwargs.get("explore", True))
        explore_noise_std = kwargs.get("explore_noise_std")
        q_guidance_probability = float(kwargs.get("q_guidance_probability", 1.0))
        action_clip = float(kwargs.get("action_clip", 1.0))
        candidate_count = kwargs.get("candidate_count")
        deterministic_eval = bool(kwargs.get("deterministic_eval", False))
        deterministic_seed = kwargs.get("deterministic_seed")
        joint_seed = None
        if deterministic_eval:
            joint_seed = 0 if deterministic_seed is None else int(deterministic_seed)

        joint_action, joint_value = self.wavegrad_catch.choose_action(
            obs=(image, robot_state),
            x_graph=graph_state,
            safety_features=safety_features,
            explore=explore,
            explore_noise_std=explore_noise_std,
            q_guidance_probability=q_guidance_probability,
            action_clip=action_clip,
            candidate_count=candidate_count,
            deterministic_seed=joint_seed,
        )
        joint_action = np.asarray(joint_action, dtype=np.float32).reshape(-1)
        if joint_action.size < 2:
            joint_action = np.pad(joint_action, (0, 2 - joint_action.size), mode="constant")
        shoulder_action = float(joint_action[0])
        arm_action = float(joint_action[1])
        catch_info = self._action_info(self.wavegrad_catch)
        return {
            "shoulder_action": shoulder_action,
            "shoulder_value": float(joint_value),
            "arm_action": arm_action,
            "arm_value": float(joint_value),
            "q_guided_used": bool(catch_info["q_guided_used"]),
            "q_guided_action_delta": float(catch_info["q_guided_action_delta"]),
            "critic_updates": int(catch_info["critic_updates"]),
            "shoulder_action_info": dict(catch_info),
            "arm_action_info": dict(catch_info),
            "catch_action_info": catch_info,
        }

    def store_catch(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        self._require_agents()
        assert self.wavegrad_catch is not None

        actions = list(kwargs.get("actions", []))
        values = list(kwargs.get("values", []))
        if len(actions) != 2 or len(values) != 2:
            raise ValueError("store_catch expects two actions and two values")

        state = self._state_triplet(kwargs.get("state"))
        next_state = self._state_triplet(kwargs.get("next_state"))
        reward = float(kwargs.get("reward", 0.0))
        done = int(kwargs.get("done", 0))
        success_flag = bool(kwargs.get("success_flag", False))
        safety_penalty = float(kwargs.get("safety_penalty", 0.0))
        safety_features = kwargs.get("safety_features")
        if safety_features is not None:
            safety_features = self._as_float_array(safety_features, np.float32)

        joint_value = float(np.mean(np.asarray(values, dtype=np.float32)))
        self.wavegrad_catch.store_transition_catch(
            state=state,
            action=[float(actions[0]), float(actions[1])],
            reward=reward,
            next_state=next_state,
            done=done,
            value=joint_value,
            safety_features=safety_features,
            success_flag=success_flag,
            safety_penalty=safety_penalty,
        )
        memory_size = len(self.wavegrad_catch.actions)
        return {
            "catch_memory_size": memory_size,
            "shoulder_memory_size": memory_size,
            "arm_memory_size": memory_size,
        }

    def learn_catch(self) -> Dict[str, Any]:
        self._require_agents()
        assert self.wavegrad_catch is not None

        loss = self.wavegrad_catch.learn()
        return self._combined_loss_result([self.wavegrad_catch], [loss], role_prefixes=("catch",))

    def choose_tai(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        self._require_agents()
        assert self.wavegrad_tai_leg_upper is not None
        assert self.wavegrad_tai_leg_lower is not None
        assert self.wavegrad_tai_ankle is not None

        image = self._as_float_array(kwargs.get("image"), np.float32)
        robot_state = self._as_float_array(kwargs.get("robot_state"), np.float32)
        graph_state = self._as_float_array(kwargs.get("graph_state", robot_state), np.float32)
        explore = bool(kwargs.get("explore", True))
        explore_noise_std = kwargs.get("explore_noise_std")
        q_guidance_probability = float(kwargs.get("q_guidance_probability", 1.0))
        action_clip = float(kwargs.get("action_clip", 1.0))
        candidate_count = kwargs.get("candidate_count")

        leg_upper_action, leg_upper_value = self.wavegrad_tai_leg_upper.choose_action(
            obs=(image, robot_state),
            x_graph=graph_state,
            explore=explore,
            explore_noise_std=explore_noise_std,
            q_guidance_probability=q_guidance_probability,
            action_clip=action_clip,
            candidate_count=candidate_count,
        )
        leg_lower_action, leg_lower_value = self.wavegrad_tai_leg_lower.choose_action(
            obs=(image, robot_state),
            x_graph=graph_state,
            explore=explore,
            explore_noise_std=explore_noise_std,
            q_guidance_probability=q_guidance_probability,
            action_clip=action_clip,
            candidate_count=candidate_count,
        )
        ankle_action, ankle_value = self.wavegrad_tai_ankle.choose_action(
            obs=(image, robot_state),
            x_graph=graph_state,
            explore=explore,
            explore_noise_std=explore_noise_std,
            q_guidance_probability=q_guidance_probability,
            action_clip=action_clip,
            candidate_count=candidate_count,
        )
        upper_info = self._action_info(self.wavegrad_tai_leg_upper)
        lower_info = self._action_info(self.wavegrad_tai_leg_lower)
        ankle_info = self._action_info(self.wavegrad_tai_ankle)
        return {
            "leg_upper_action": float(leg_upper_action),
            "leg_upper_value": float(leg_upper_value),
            "leg_lower_action": float(leg_lower_action),
            "leg_lower_value": float(leg_lower_value),
            "ankle_action": float(ankle_action),
            "ankle_value": float(ankle_value),
            "q_guided_used": bool(
                upper_info["q_guided_used"] or lower_info["q_guided_used"] or ankle_info["q_guided_used"]
            ),
            "q_guided_action_delta": float(
                (
                    upper_info["q_guided_action_delta"]
                    + lower_info["q_guided_action_delta"]
                    + ankle_info["q_guided_action_delta"]
                )
                / 3.0
            ),
            "critic_updates": int(
                upper_info["critic_updates"] + lower_info["critic_updates"] + ankle_info["critic_updates"]
            ),
            "leg_upper_action_info": upper_info,
            "leg_lower_action_info": lower_info,
            "ankle_action_info": ankle_info,
        }

    def store_tai(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        self._require_agents()
        assert self.wavegrad_tai_leg_upper is not None
        assert self.wavegrad_tai_leg_lower is not None
        assert self.wavegrad_tai_ankle is not None

        actions = list(kwargs.get("actions", []))
        values = list(kwargs.get("values", []))
        if len(actions) != 3 or len(values) != 3:
            raise ValueError("store_tai expects three actions and three values")

        state = self._state_triplet(kwargs.get("state"))
        next_state = self._state_triplet(kwargs.get("next_state"))
        reward = float(kwargs.get("reward", 0.0))
        done = int(kwargs.get("done", 0))
        success_flag = bool(kwargs.get("success_flag", False))
        safety_penalty = float(kwargs.get("safety_penalty", 0.0))

        agents = [self.wavegrad_tai_leg_upper, self.wavegrad_tai_leg_lower, self.wavegrad_tai_ankle]
        for agent, action, value in zip(agents, actions, values):
            agent.store_transition_tai(
                state=state,
                action=float(action),
                reward=reward,
                next_state=next_state,
                done=done,
                value=float(value),
                success_flag=success_flag,
                safety_penalty=safety_penalty,
            )
        return {
            "leg_upper_memory_size": len(self.wavegrad_tai_leg_upper.actions),
            "leg_lower_memory_size": len(self.wavegrad_tai_leg_lower.actions),
            "ankle_memory_size": len(self.wavegrad_tai_ankle.actions),
        }

    def learn_tai(self) -> Dict[str, Any]:
        self._require_agents()
        assert self.wavegrad_tai_leg_upper is not None
        assert self.wavegrad_tai_leg_lower is not None
        assert self.wavegrad_tai_ankle is not None

        agents = [self.wavegrad_tai_leg_upper, self.wavegrad_tai_leg_lower, self.wavegrad_tai_ankle]
        losses = [agent.learn() for agent in agents]
        return self._combined_loss_result(agents, losses, role_prefixes=("leg_upper", "leg_lower", "ankle"))

    def _combined_loss_result(
        self,
        agents: Iterable[Any],
        losses: Iterable[float],
        role_prefixes: Iterable[str],
    ) -> Dict[str, Any]:
        agents = list(agents)
        losses = [float(loss) for loss in losses]
        result: Dict[str, Any] = {
            "loss": float(sum(losses)),
            "diffusion_loss": float(sum(agent.last_loss_info.get("diffusion_loss", 0.0) for agent in agents)),
            "value_loss": float(sum(agent.last_loss_info.get("value_loss", 0.0) for agent in agents)),
            "q_loss": float(sum(agent.last_loss_info.get("q_loss", 0.0) for agent in agents)),
            "q_guidance_loss": float(sum(agent.last_loss_info.get("q_guidance_loss", 0.0) for agent in agents)),
            "q_guidance_loss_used": float(
                sum(agent.last_loss_info.get("q_guidance_loss_used", 0.0) for agent in agents)
            ),
            "q_guidance_loss_ratio": float(
                sum(agent.last_loss_info.get("q_guidance_loss_ratio", 0.0) for agent in agents)
                / max(1, len(agents))
            ),
            "success_replay_size": int(sum(agent.last_loss_info.get("success_replay_size", 0) for agent in agents)),
            "elite_replay_size": int(sum(agent.last_loss_info.get("elite_replay_size", 0) for agent in agents)),
            "q_guided_used": int(any(agent.last_loss_info.get("q_guided_used", 0) for agent in agents)),
            "q_guided_action_delta": float(
                sum(agent.last_loss_info.get("q_guided_action_delta", 0.0) for agent in agents) / max(1, len(agents))
            ),
            "critic_updates": int(sum(agent.last_loss_info.get("critic_updates", 0) for agent in agents)),
            "policy_lr": float(
                sum(agent.last_loss_info.get("policy_lr", 0.0) for agent in agents) / max(1, len(agents))
            ),
        }
        for role, agent, loss in zip(role_prefixes, agents, losses):
            role_info = dict(agent.last_loss_info)
            role_info["loss"] = float(loss)
            result["%s_loss_info" % role] = role_info
        return result

    def save_catch_checkpoint(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        self._require_agents()
        assert self.wavegrad_catch is not None

        episode = int(kwargs.get("episode", 0))
        episode_return = float(kwargs.get("episode_return", 0.0))
        success_flag = int(kwargs.get("success_flag", kwargs.get("catch_success", 0)))
        test_success_rate = float(kwargs.get("test_success_rate", 0.0))
        successful_test_episodes = int(kwargs.get("successful_test_episodes", 0))
        num_test_episodes = int(kwargs.get("num_test_episodes", 0))
        is_new_best = test_success_rate > self.best_catch_test_success_rate
        if is_new_best:
            self.best_catch_test_success_rate = test_success_rate

        checkpoint = {
            "wavegrad_catch": self.wavegrad_catch.policy.state_dict(),
            "optimizer_wavegrad_catch": self.wavegrad_catch.optimizer.state_dict(),
            "critic_wavegrad_catch": self.wavegrad_catch.critic.state_dict(),
            "optimizer_critic_wavegrad_catch": self.wavegrad_catch.critic_optimizer.state_dict(),
            "replay_wavegrad_catch": self.wavegrad_catch.export_replay_state(max_items=64),
            "catch_action_dim": int(self.wavegrad_catch.action_dim),
            "episode": episode,
            "episode_return": episode_return,
            "catch_success": success_flag,
            "test_success_rate": test_success_rate,
            "successful_test_episodes": successful_test_episodes,
            "num_test_episodes": num_test_episodes,
            "rank_score": test_success_rate,
            "best_test_success_rate": self.best_catch_test_success_rate,
        }

        base_dir = path_list["model_path_catch_WaveGrad"]
        _ensure_dir(base_dir)
        save_path = os.path.join(base_dir, "wavegrad_model_%d.ckpt" % episode)
        torch.save(checkpoint, save_path)
        ranked_path = self.model_ranking.add_and_manage(
            new_score=test_success_rate,
            new_checkpoint=checkpoint,
            episode_id=episode,
            base_dir=base_dir,
            success_count=successful_test_episodes,
        )
        best_path = None
        if is_new_best:
            best_path = os.path.join(base_dir, "best_catch_WaveGrad.pth")
            torch.save(checkpoint, best_path)
            print("Saved best WaveGrad catch checkpoint: %s" % best_path, flush=True)
        print("Saved WaveGrad catch checkpoint: %s" % save_path, flush=True)
        return {"save_path": save_path, "ranked_path": ranked_path, "best_path": best_path}

    def save_tai_checkpoint(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        self._require_agents()
        assert self.wavegrad_tai_leg_upper is not None
        assert self.wavegrad_tai_leg_lower is not None
        assert self.wavegrad_tai_ankle is not None

        total_episode = int(kwargs.get("total_episode", 0))
        tai_episode = int(kwargs.get("tai_episode", kwargs.get("episode", 0)))
        checkpoint = {
            "episode": tai_episode,
            "wavegrad_tai_leg_upper": self.wavegrad_tai_leg_upper.policy.state_dict(),
            "optimizer_wavegrad_tai_leg_upper": self.wavegrad_tai_leg_upper.optimizer.state_dict(),
            "critic_wavegrad_tai_leg_upper": self.wavegrad_tai_leg_upper.critic.state_dict(),
            "optimizer_critic_wavegrad_tai_leg_upper": self.wavegrad_tai_leg_upper.critic_optimizer.state_dict(),
            "replay_wavegrad_tai_leg_upper": self.wavegrad_tai_leg_upper.export_replay_state(max_items=64),
            "wavegrad_tai_leg_lower": self.wavegrad_tai_leg_lower.policy.state_dict(),
            "optimizer_wavegrad_tai_leg_lower": self.wavegrad_tai_leg_lower.optimizer.state_dict(),
            "critic_wavegrad_tai_leg_lower": self.wavegrad_tai_leg_lower.critic.state_dict(),
            "optimizer_critic_wavegrad_tai_leg_lower": self.wavegrad_tai_leg_lower.critic_optimizer.state_dict(),
            "replay_wavegrad_tai_leg_lower": self.wavegrad_tai_leg_lower.export_replay_state(max_items=64),
            "wavegrad_tai_ankle": self.wavegrad_tai_ankle.policy.state_dict(),
            "optimizer_wavegrad_tai_ankle": self.wavegrad_tai_ankle.optimizer.state_dict(),
            "critic_wavegrad_tai_ankle": self.wavegrad_tai_ankle.critic.state_dict(),
            "optimizer_critic_wavegrad_tai_ankle": self.wavegrad_tai_ankle.critic_optimizer.state_dict(),
            "replay_wavegrad_tai_ankle": self.wavegrad_tai_ankle.export_replay_state(max_items=64),
        }
        base_dir = path_list["model_path_tai_WaveGrad"]
        _ensure_dir(base_dir)
        save_path = os.path.join(base_dir, "wavegrad_model_tai_%d_%d.ckpt" % (total_episode, tai_episode))
        torch.save(checkpoint, save_path)
        print("Saved WaveGrad tai checkpoint: %s" % save_path, flush=True)
        return {"save_path": save_path}

    def set_mode(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        self._require_agents()
        stage = str(kwargs.get("stage", "catch")).lower()
        mode = str(kwargs.get("mode", "train")).lower()
        agents: List[Any] = []
        if stage in ("catch", "all"):
            agents.append(self.wavegrad_catch)
        if stage in ("tai", "all"):
            agents.extend([self.wavegrad_tai_leg_upper, self.wavegrad_tai_leg_lower, self.wavegrad_tai_ankle])
        for agent in agents:
            if agent is None:
                continue
            if mode == "eval":
                agent.policy.eval()
                if hasattr(agent, "critic"):
                    agent.critic.eval()
            else:
                agent.policy.train()
                if hasattr(agent, "critic"):
                    agent.critic.train()
        return {"stage": stage, "mode": mode}

    def dispatch(self, cmd: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if cmd == "init":
            return self.init_agents(
                model_path=kwargs.get("model_path"),
                max_steps_per_episode=int(kwargs.get("max_steps_per_episode", 22)),
            )
        if cmd == "runtime_info":
            return self.runtime_info()
        if cmd == "choose_catch":
            return self.choose_catch(kwargs)
        if cmd == "store_catch":
            return self.store_catch(kwargs)
        if cmd == "learn_catch":
            return self.learn_catch()
        if cmd == "choose_tai":
            return self.choose_tai(kwargs)
        if cmd == "store_tai":
            return self.store_tai(kwargs)
        if cmd == "learn_tai":
            return self.learn_tai()
        if cmd == "save_catch_checkpoint":
            return self.save_catch_checkpoint(kwargs)
        if cmd == "save_tai_checkpoint":
            return self.save_tai_checkpoint(kwargs)
        if cmd == "set_mode":
            return self.set_mode(kwargs)
        if cmd == "close":
            self.running = False
            return {"closed": True}
        raise ValueError("unknown command: %s" % cmd)

    def serve_forever(self) -> None:
        print("WaveGrad GPU service starting on %s:%d" % (self.host, self.port), flush=True)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((self.host, self.port))
            server_sock.listen(1)
            print("WaveGrad GPU service listening on %s:%d" % (self.host, self.port), flush=True)
            while self.running:
                conn, addr = server_sock.accept()
                print("WaveGrad GPU client connected: %s:%s" % addr, flush=True)
                with conn:
                    while self.running:
                        try:
                            request = recv_packet(conn)
                        except EOFError:
                            break

                        cmd = str(request.get("cmd", ""))
                        kwargs = request.get("kwargs", {}) or {}
                        try:
                            result = self.dispatch(cmd, kwargs)
                            send_packet(conn, {"ok": True, "result": result})
                        except Exception as exc:
                            send_packet(
                                conn,
                                {
                                    "ok": False,
                                    "error": "%s: %s" % (type(exc).__name__, exc),
                                    "traceback": traceback.format_exc(),
                                },
                            )
                print("WaveGrad GPU client disconnected.", flush=True)


def _once_smoke(host: str, port: int) -> None:
    service = WaveGradGPUService(host=host, port=port)
    info = service.init_agents(max_steps_per_episode=3)
    image = np.zeros((128, 128), dtype=np.float32)
    robot_state = np.zeros(20, dtype=np.float32)
    graph_state = np.zeros(19, dtype=np.float32)
    safety_features = np.zeros(14, dtype=np.float32)
    choice = service.choose_catch(
        {
            "image": image,
            "robot_state": robot_state,
            "graph_state": graph_state,
            "safety_features": safety_features,
            "explore": False,
        }
    )
    print("once-smoke runtime:", info, flush=True)
    print("once-smoke choose_catch:", choice, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WaveGrad GPU algorithm service")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--once-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.once_smoke:
        _once_smoke(args.host, args.port)
        return
    WaveGradGPUService(host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
