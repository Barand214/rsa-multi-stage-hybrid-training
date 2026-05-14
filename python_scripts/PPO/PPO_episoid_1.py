# 测试
import glob
import heapq
import os
import re

import numpy as np
import torch

from python_scripts.PPO.PPO_PPOnet import PPO
from python_scripts.PPO.PPO_PPOnet_2 import PPO2
from python_scripts.PPO.PPO_episoid_2_1 import PPO_tai_episoid
from python_scripts.PPO_Log_write import Log_write
from python_scripts.Project_config import path_list
from python_scripts.Webots_interfaces import Environment
from python_scripts.utils.sensor_utils import reset_environment, wait_for_sensors_stable


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _next_log_file(base_dir, prefix):
    _ensure_dir(base_dir)
    pattern = os.path.join(base_dir, f"{prefix}_*.json")
    existing_logs = glob.glob(pattern)
    latest_num = 0
    for log_path in existing_logs:
        match = re.search(rf"{re.escape(prefix)}_(\d+)", os.path.basename(log_path))
        if match:
            latest_num = max(latest_num, int(match.group(1)))
    return os.path.join(base_dir, f"{prefix}_{latest_num + 1}.json")


def _extract_last_number(file_path):
    numbers = re.findall(r"\d+", os.path.basename(file_path))
    return int(numbers[-1]) if numbers else 0


def _to_scalar(action):
    arr = np.asarray(action).reshape(-1)
    if arr.size == 0:
        return float(action)
    return float(arr[0])


def _load_catch_model(model_path, ppo_shoulder, ppo_arm):
    target_model = model_path

    if not target_model:
        model_files = glob.glob(os.path.join(path_list["model_path_catch_PPO"], "ppo_model_*.ckpt"))
        if not model_files:
            model_files = glob.glob(os.path.join(path_list["model_path_catch_PPO"], "ppo_model_success_*.ckpt"))

        if not model_files:
            print("未找到已保存的抓取模型，从头开始训练")
            return 0

        target_model = max(model_files, key=_extract_last_number)

    if not os.path.isfile(target_model):
        print(f"指定抓取模型不存在: {target_model}，从头开始训练")
        return 0

    try:
        checkpoint = torch.load(target_model, map_location="cpu")
        if isinstance(checkpoint, dict) and "policy_shoulder" in checkpoint:
            ppo_shoulder.policy.load_state_dict(checkpoint["policy_shoulder"], strict=False)
            ppo_arm.policy.load_state_dict(checkpoint["policy_arm"], strict=False)
            if "optimizer_shoulder" in checkpoint:
                ppo_shoulder.optimizer.load_state_dict(checkpoint["optimizer_shoulder"])
            if "optimizer_arm" in checkpoint:
                ppo_arm.optimizer.load_state_dict(checkpoint["optimizer_arm"])
            episode_start = int(checkpoint.get("episode", _extract_last_number(target_model)))
        else:
            # 兼容旧格式：一个 state_dict 同时加载到两个策略
            ppo_shoulder.policy.load_state_dict(checkpoint, strict=False)
            ppo_arm.policy.load_state_dict(checkpoint, strict=False)
            episode_start = _extract_last_number(target_model)

        print(f"加载抓取模型: {target_model}，从周期 {episode_start} 继续训练")
        return episode_start
    except Exception as e:
        print(f"抓取模型加载失败: {e}")
        return 0


def _parse_tai_episode(file_path):
    # 兼容 ppo_model_tai_{total_episode}_{episode}.ckpt
    match = re.search(r"ppo_model_tai_(\d+)_(\d+)\.ckpt", os.path.basename(file_path))
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, _extract_last_number(file_path)


def _load_tai_model(ppo2_leg_upper, ppo2_leg_lower, ppo2_ankle, default_episode=1):
    model_files = glob.glob(os.path.join(path_list["model_path_tai_PPO"], "ppo_model_tai_*.ckpt"))
    if not model_files:
        print("未找到已保存的抬腿模型，从头开始训练")
        return default_episode

    latest_model = max(model_files, key=_parse_tai_episode)
    total_ep, ep = _parse_tai_episode(latest_model)

    try:
        checkpoint = torch.load(latest_model, map_location="cpu")
        if isinstance(checkpoint, dict) and "policy_LegUpper" in checkpoint:
            ppo2_leg_upper.policy.load_state_dict(checkpoint["policy_LegUpper"], strict=False)
            ppo2_leg_lower.policy.load_state_dict(checkpoint["policy_LegLower"], strict=False)
            ppo2_ankle.policy.load_state_dict(checkpoint["policy_Ankle"], strict=False)

            if "optimizer_LegUpper" in checkpoint:
                ppo2_leg_upper.optimizer.load_state_dict(checkpoint["optimizer_LegUpper"])
            if "optimizer_LegLower" in checkpoint:
                ppo2_leg_lower.optimizer.load_state_dict(checkpoint["optimizer_LegLower"])
            if "optimizer_Ankle" in checkpoint:
                ppo2_ankle.optimizer.load_state_dict(checkpoint["optimizer_Ankle"])

            tai_episode = int(checkpoint.get("episode", ep))
        else:
            ppo2_leg_upper.policy.load_state_dict(checkpoint, strict=False)
            ppo2_leg_lower.policy.load_state_dict(checkpoint, strict=False)
            ppo2_ankle.policy.load_state_dict(checkpoint, strict=False)
            tai_episode = ep

        print(f"加载抬腿模型: {latest_model}，总周期 {total_ep}，抬腿周期 {tai_episode}")
        return max(default_episode, tai_episode)
    except Exception as e:
        print(f"抬腿模型加载失败: {e}")
        return default_episode


def _is_catch_success(env):
    left_any = any(
        [
            env.get_touch_sensor_value("grasp_L1"),
            env.get_touch_sensor_value("grasp_L1_1"),
            env.get_touch_sensor_value("grasp_L1_2"),
        ]
    )
    right_any = any(
        [
            env.get_touch_sensor_value("grasp_R1"),
            env.get_touch_sensor_value("grasp_R1_1"),
            env.get_touch_sensor_value("grasp_R1_2"),
        ]
    )
    return bool(left_any and right_any)


def _step_catch(env, action_shoulder, action_arm, steps):
    """按当前 RobotRun1 接口执行抓取一步。"""
    from python_scripts.PPO.RobotRun1 import RobotRun

    discrete_action = [1, 1]
    continuous_action = [float(action_shoulder), float(action_arm)]
    next_state, reward, done, catch_success = RobotRun(
        env.robot,
        discrete_action,
        continuous_action,
        steps,
    ).run()
    return next_state, float(reward), int(done), bool(catch_success)


class ModelRanking:
    """维护抓取模型排行榜，仅保留前 N 个模型文件。"""

    def __init__(self, top_n=5, key_name="episode_return"):
        self.top_n = top_n
        self.key_name = key_name
        self.rankings = []  # min-heap: (score, path)

    def add_and_manage(self, new_score, new_checkpoint, episode_id, base_dir, filename_prefix="ppo_model"):
        new_score = float(new_score)
        should_save = False
        final_save_path = ""

        if len(self.rankings) < self.top_n:
            should_save = True
            final_save_path = os.path.join(base_dir, f"{filename_prefix}_{episode_id}.ckpt")
        elif new_score > self.rankings[0][0]:
            should_save = True
            final_save_path = os.path.join(base_dir, f"{filename_prefix}_{episode_id}.ckpt")
            worst_score, worst_path = heapq.heappop(self.rankings)
            try:
                os.remove(worst_path)
                print(f"删除旧模型文件: {worst_path} ({self.key_name}: {worst_score:.4f})")
            except FileNotFoundError:
                print(f"警告: 试图删除不存在的文件 {worst_path}")

        if not should_save:
            print(
                f"模型 {episode_id} ({self.key_name}: {new_score:.4f}) 未进入前 {self.top_n}，不保存。"
            )
            return None

        torch.save(new_checkpoint, final_save_path)
        heapq.heappush(self.rankings, (new_score, final_save_path))
        print(
            f"模型 {episode_id} ({self.key_name}: {new_score:.4f}) 已保存并进入前 {self.top_n}: {final_save_path}"
        )
        return final_save_path

    def print_current_rankings(self):
        if not self.rankings:
            print("当前排行榜为空。")
            return

        print(f"\n--- 抓取模型排行榜 (Top {self.top_n}) ---")
        sorted_rankings = sorted(self.rankings, key=lambda x: x[0], reverse=True)
        for i, (score, path) in enumerate(sorted_rankings, 1):
            print(f"  {i}. {self.key_name}: {score:.4f}, Path: {path}")
        print("-----------------------------------------\n")


def PPO_episoid_1(model_path=None, max_steps_per_episode=500):
    # 抓取智能体（两个独立 PPO）
    ppo_shoulder = PPO(node_num=19, env_information=None)
    ppo_arm = PPO(node_num=19, env_information=None)

    # 抬腿智能体（沿用现有 PPO2 三关节结构）
    ppo2_leg_upper = PPO2(node_num=19, env_information=None)
    ppo2_leg_lower = PPO2(node_num=19, env_information=None)
    ppo2_ankle = PPO2(node_num=19, env_information=None)

    log_writer_catch = Log_write()
    log_writer_tai = Log_write()
    model_ranking = ModelRanking(top_n=5, key_name="episode_return")

    catch_checkpoint_dir = path_list["model_path_catch_PPO"]
    tai_checkpoint_dir = path_list["model_path_tai_PPO"]
    _ensure_dir(catch_checkpoint_dir)
    _ensure_dir(tai_checkpoint_dir)
    _ensure_dir(path_list["catch_log_path_PPO"])
    _ensure_dir(path_list["tai_log_path_PPO"])

    log_file_latest_catch = _next_log_file(path_list["catch_log_path_PPO"], "catch_log")
    log_file_latest_tai = _next_log_file(path_list["tai_log_path_PPO"], "tai_log")
    print(f"将使用新的抓取日志目录: {log_file_latest_catch}")
    print(f"将使用新的抬腿日志目录: {log_file_latest_tai}")

    episode_start = _load_catch_model(model_path, ppo_shoulder, ppo_arm)
    tai_episode = _load_tai_model(
        ppo2_leg_upper,
        ppo2_leg_lower,
        ppo2_ankle,
        default_episode=1,
    )

    env = Environment()

    for episode in range(episode_start, episode_start + 10000):
        print(f"\n==============================")
        print(f"Catch Episode {episode}")
        print(f"==============================")

        env.reset()
        env.wait(500)
        if not wait_for_sensors_stable(env, max_retries=40, wait_ms=200):
            print("警告: 传感器不稳定，尝试重置环境...")
            reset_environment(env)

        steps = 0
        done = 0
        episode_return = 0.0
        catch_success = False
        imgs = []

        log_writer_catch.add(episode_num=episode)

        while True:
            obs_img, obs_tensor = env.get_img(steps, imgs)
            robot_state = env.get_robot_state()
            obs = (obs_tensor, robot_state)

            action_shoulder_raw, log_prob_shoulder, value_shoulder = ppo_shoulder.choose_action(
                episode_num=episode,
                obs=obs,
                x_graph=robot_state,
                action_type="shoulder",
            )
            action_arm_raw, log_prob_arm, value_arm = ppo_arm.choose_action(
                episode_num=episode,
                obs=obs,
                x_graph=robot_state,
                action_type="arm",
            )

            action_shoulder = _to_scalar(action_shoulder_raw)
            action_arm = _to_scalar(action_arm_raw)

            next_state, reward, done, step_catch_success = _step_catch(
                env,
                action_shoulder,
                action_arm,
                steps,
            )

            next_obs_img, _ = env.get_img(steps + 1, imgs)

            ppo_shoulder.store_transition_catch(
                state=[obs_img, robot_state, robot_state],
                action_shoulder=action_shoulder,
                action_arm=action_arm,
                reward=reward,
                next_state=[next_obs_img, next_state, next_state],
                done=done,
                value_shoulder=value_shoulder,
                value_arm=value_arm,
                log_prob_shoulder=log_prob_shoulder,
                log_prob_arm=log_prob_arm,
            )
            ppo_arm.store_transition_catch(
                state=[obs_img, robot_state, robot_state],
                action_shoulder=action_shoulder,
                action_arm=action_arm,
                reward=reward,
                next_state=[next_obs_img, next_state, next_state],
                done=done,
                value_shoulder=value_shoulder,
                value_arm=value_arm,
                log_prob_shoulder=log_prob_shoulder,
                log_prob_arm=log_prob_arm,
            )

            log_writer_catch.add_action_catch(action_shoulder, action_arm)
            episode_return += float(reward)
            catch_success = bool(catch_success or step_catch_success or _is_catch_success(env))

            steps += 1
            if done == 1 or steps >= max_steps_per_episode:
                break

        loss_shoulder = ppo_shoulder.learn(action_type="shoulder")
        loss_arm = ppo_arm.learn(action_type="arm")

        log_writer_catch.add(loss=loss_shoulder + loss_arm)
        log_writer_catch.add(return_all=episode_return)
        log_writer_catch.add(goal=1 if catch_success else 0)
        log_writer_catch.clear()
        log_writer_catch.save_catch(log_file_latest_catch)

        if episode > 0 and episode % 500 == 0:
            checkpoint = {
                "policy_shoulder": ppo_shoulder.policy.state_dict(),
                "optimizer_shoulder": ppo_shoulder.optimizer.state_dict(),
                "policy_arm": ppo_arm.policy.state_dict(),
                "optimizer_arm": ppo_arm.optimizer.state_dict(),
                "episode": episode,
                "episode_return": episode_return,
                "catch_success": int(catch_success),
                "rank_score": float(episode_return),
            }
            model_ranking.add_and_manage(
                new_score=episode_return,
                new_checkpoint=checkpoint,
                episode_id=episode,
                base_dir=catch_checkpoint_dir,
                filename_prefix="ppo_model",
            )
            model_ranking.print_current_rankings()

        if catch_success:
            print("抓取成功，进入抬腿训练...")
            PPO_tai_episoid(
                ppo2_LegUpper=ppo2_leg_upper,
                ppo2_LegLower=ppo2_leg_lower,
                ppo2_Ankle=ppo2_ankle,
                existing_env=env,
                total_episode=episode,
                episode=tai_episode,
                log_writer_tai=log_writer_tai,
                log_file_latest_tai=log_file_latest_tai,
            )
            tai_episode += 1
            env.reset()
            env.wait(500)

    return False, env
