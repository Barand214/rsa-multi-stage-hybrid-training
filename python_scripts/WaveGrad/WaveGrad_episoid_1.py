import glob
import os
import re

import numpy as np

from python_scripts.WaveGrad.WaveGrad_episoid_2_1 import WaveGrad_tai_episoid
from python_scripts.WaveGrad.wavegrad_gpu_client import WaveGradGPUClient
from python_scripts.WaveGrad.WaveGrad_log_write import Log_write
from python_scripts.Project_config import Darwin_config, gps_goal1, path_list
from python_scripts.utils.sensor_utils import reset_environment, wait_for_sensors_stable


CHECKPOINT_INTERVAL = 500
NUM_TEST_EPISODES = 200
MAX_TEST_ATTEMPTS = 5
MAX_TEST_INIT_FAILURES = 25


def _catch_train_schedule(episode):
    if episode < 300:
        return {
            "explore_noise_std": 0.06,
            "action_clip": 0.85,
        }
    if episode < 1000:
        return {
            "explore_noise_std": 0.04,
            "action_clip": 0.85,
        }
    if episode < 2000:
        return {
            "explore_noise_std": 0.02,
            "action_clip": 0.85,
        }
    return {
        "explore_noise_std": 0.005,
        "action_clip": 0.85,
    }


def _catch_eval_schedule():
    return {
        "explore_noise_std": 0.0,
        "action_clip": 0.85,
    }


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _next_log_file(base_dir, prefix):
    _ensure_dir(base_dir)
    pattern = os.path.join(base_dir, f"{prefix}_*.json")
    latest_num = 0
    for log_path in glob.glob(pattern):
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


def _clip_delta_to_limits(robot_state, left_idx, right_idx, desired_delta, max_delta):
    left_low, left_high = Darwin_config.limit[left_idx]
    right_low, right_high = Darwin_config.limit[right_idx]
    left_pos = robot_state[left_idx]
    right_pos = robot_state[right_idx]
    margin = 0.02

    lower = max(-max_delta, left_low + margin - left_pos, right_pos - (right_high - margin))
    upper = min(max_delta, left_high - margin - left_pos, right_pos - (right_low + margin))
    if lower > upper:
        lower = max(-max_delta, left_low - left_pos, right_pos - right_high)
        upper = min(max_delta, left_high - left_pos, right_pos - right_low)
    if lower > upper:
        return 0.0, True
    return float(np.clip(desired_delta, lower, upper)), False


def _imu_margin_ratio(env):
    try:
        acc = env.darwin.accelerometer.getValues()
        gyro = env.darwin.gyro.getValues()
    except Exception:
        return 1.0

    ratios = []
    for values, lows, highs in (
        (acc, Darwin_config.acc_low, Darwin_config.acc_high),
        (gyro, Darwin_config.gyro_low, Darwin_config.gyro_high),
    ):
        for value, low, high in zip(values, lows, highs):
            width = high - low
            if width <= 0:
                continue
            ratios.append(min((value - low) / width, (high - value) / width))
    return min(ratios) if ratios else 1.0


def _joint_margin_penalty(future_state):
    penalty = 0.0
    for idx in (0, 1, 4, 5):
        low, high = Darwin_config.limit[idx]
        margin = min(future_state[idx] - low, high - future_state[idx])
        if margin < 0:
            penalty += 10.0
        elif margin < 0.04:
            penalty += (0.04 - margin) * 20.0
    return penalty


def _normalize_margin(value):
    return float(np.clip(value / 0.5, -1.0, 1.0))


def _build_safety_features(env, robot_state, prev_action_shoulder, prev_action_arm, steps, max_steps_per_episode):
    features = []
    for idx in (1, 0, 5, 4):
        low, high = Darwin_config.limit[idx]
        pos = robot_state[idx] if idx < len(robot_state) else 0.0
        features.append(_normalize_margin(pos - low))
        features.append(_normalize_margin(high - pos))

    imu_margin = _imu_margin_ratio(env)
    features.append(float(np.clip(imu_margin, -1.0, 1.0)))
    features.append(float(np.clip(prev_action_shoulder, -1.0, 1.0)))
    features.append(float(np.clip(prev_action_arm, -1.0, 1.0)))
    features.append(float(np.clip(steps / max(1, max_steps_per_episode - 1), 0.0, 1.0)))

    if len(robot_state) >= 6:
        shoulder_room = min(
            Darwin_config.limit[1][1] - robot_state[1],
            robot_state[1] - Darwin_config.limit[1][0],
            Darwin_config.limit[0][1] - robot_state[0],
            robot_state[0] - Darwin_config.limit[0][0],
        )
        arm_room = min(
            Darwin_config.limit[5][1] - robot_state[5],
            robot_state[5] - Darwin_config.limit[5][0],
            Darwin_config.limit[4][1] - robot_state[4],
            robot_state[4] - Darwin_config.limit[4][0],
        )
    else:
        shoulder_room = 0.0
        arm_room = 0.0
    features.append(_normalize_margin(shoulder_room))
    features.append(_normalize_margin(arm_room))
    return np.asarray(features, dtype=np.float32)


def _apply_action_safety_filter(env, robot_state, action_shoulder, action_arm):
    if len(robot_state) < 6:
        return action_shoulder, action_arm, 0.0, {"clipped": False, "imu_margin": 1.0}

    imu_margin = _imu_margin_ratio(env)
    max_delta = 0.14 if imu_margin < 0.12 else 0.22
    action_limit = 0.85

    action_shoulder = float(np.clip(action_shoulder, -action_limit, action_limit))
    action_arm = float(np.clip(action_arm, -action_limit, action_limit))

    desired_shoulder_delta = 0.2995 * action_shoulder - 0.145 - robot_state[1]
    desired_arm_delta = 1.25 * action_arm + 0.25 - robot_state[5]
    desired_shoulder_delta = float(np.clip(desired_shoulder_delta, -0.3, 0.3))
    desired_arm_delta = float(np.clip(desired_arm_delta, -0.3, 0.3))

    safe_shoulder_delta, shoulder_infeasible = _clip_delta_to_limits(
        robot_state, left_idx=1, right_idx=0, desired_delta=desired_shoulder_delta, max_delta=max_delta
    )
    safe_arm_delta, arm_infeasible = _clip_delta_to_limits(
        robot_state, left_idx=5, right_idx=4, desired_delta=desired_arm_delta, max_delta=max_delta
    )

    safe_action_shoulder = (robot_state[1] + safe_shoulder_delta + 0.145) / 0.2995
    safe_action_arm = (robot_state[5] + safe_arm_delta - 0.25) / 1.25
    safe_action_shoulder = float(np.clip(safe_action_shoulder, -action_limit, action_limit))
    safe_action_arm = float(np.clip(safe_action_arm, -action_limit, action_limit))

    executed_shoulder_delta = 0.2995 * safe_action_shoulder - 0.145 - robot_state[1]
    executed_arm_delta = 1.25 * safe_action_arm + 0.25 - robot_state[5]
    executed_shoulder_delta = float(np.clip(executed_shoulder_delta, -0.3, 0.3))
    executed_arm_delta = float(np.clip(executed_arm_delta, -0.3, 0.3))
    executed_shoulder_delta, shoulder_infeasible_2 = _clip_delta_to_limits(
        robot_state, left_idx=1, right_idx=0, desired_delta=executed_shoulder_delta, max_delta=max_delta
    )
    executed_arm_delta, arm_infeasible_2 = _clip_delta_to_limits(
        robot_state, left_idx=5, right_idx=4, desired_delta=executed_arm_delta, max_delta=max_delta
    )
    safe_action_shoulder = float(np.clip((robot_state[1] + executed_shoulder_delta + 0.145) / 0.2995, -action_limit, action_limit))
    safe_action_arm = float(np.clip((robot_state[5] + executed_arm_delta - 0.25) / 1.25, -action_limit, action_limit))

    future_state = list(robot_state)
    future_state[1] = robot_state[1] + executed_shoulder_delta
    future_state[0] = robot_state[0] - executed_shoulder_delta
    future_state[5] = robot_state[5] + executed_arm_delta
    future_state[4] = robot_state[4] - executed_arm_delta

    delta_adjustment = abs(desired_shoulder_delta - executed_shoulder_delta) + abs(desired_arm_delta - executed_arm_delta)
    imu_penalty = max(0.0, 0.12 - imu_margin) * 8.0
    safety_penalty = delta_adjustment * 6.0 + _joint_margin_penalty(future_state) + imu_penalty
    if shoulder_infeasible or arm_infeasible or shoulder_infeasible_2 or arm_infeasible_2:
        safety_penalty += 5.0

    clipped = (
        abs(safe_action_shoulder - action_shoulder) > 1e-6
        or abs(safe_action_arm - action_arm) > 1e-6
    )
    info = {
        "clipped": clipped,
        "imu_margin": imu_margin,
        "max_delta": max_delta,
        "shoulder_delta": executed_shoulder_delta,
        "arm_delta": executed_arm_delta,
        "infeasible": shoulder_infeasible or arm_infeasible or shoulder_infeasible_2 or arm_infeasible_2,
    }
    return float(safe_action_shoulder), float(safe_action_arm), float(safety_penalty), info


def _get_grasp_contact(env):
    all_grasp_sensors = [
        env.darwin.get_touch_sensor_value("grasp_L1"),
        env.darwin.get_touch_sensor_value("grasp_L1_1"),
        env.darwin.get_touch_sensor_value("grasp_L1_2"),
        env.darwin.get_touch_sensor_value("grasp_R1"),
        env.darwin.get_touch_sensor_value("grasp_R1_1"),
        env.darwin.get_touch_sensor_value("grasp_R1_2"),
    ]
    left_any = any(all_grasp_sensors[0:3])
    right_any = any(all_grasp_sensors[3:6])
    return all_grasp_sensors, left_any, right_any


def _step_catch(env, action_shoulder, action_arm, steps):
    from python_scripts.WaveGrad.RobotRun1 import RobotRun

    discrete_action = [1, 1]
    continuous_action = [float(action_shoulder), float(action_arm)]
    next_state, env_reward, done, catch_success = RobotRun(
        env.robot,
        discrete_action,
        continuous_action,
        steps,
    ).run()
    return next_state, float(env_reward), int(done), bool(catch_success)


def _run_wavegrad_catch_tests(env, gpu, checkpoint_episode, max_steps_per_test_episode):
    print(f"\n--- Episode {checkpoint_episode}: testing WaveGrad catch model ({NUM_TEST_EPISODES} valid runs) ---")
    gpu.set_mode(stage="catch", mode="eval")
    successful_test_episodes = 0
    valid_test_cnt = 0
    total_test_cnt = 0
    consecutive_init_failures = 0

    try:
        while valid_test_cnt < NUM_TEST_EPISODES:
            total_test_cnt += 1
            is_test_valid = False
            for init_try in range(MAX_TEST_ATTEMPTS):
                env.reset()
                env.wait(200)
                if wait_for_sensors_stable(env, max_retries=40, wait_ms=200):
                    is_test_valid = True
                    break
                print(f"  Sensor init retry {init_try + 1}/{MAX_TEST_ATTEMPTS}")
            if not is_test_valid:
                consecutive_init_failures += 1
                print(
                    "  Test init failed; skipping this run "
                    f"({consecutive_init_failures}/{MAX_TEST_INIT_FAILURES} consecutive failures)."
                )
                if consecutive_init_failures >= MAX_TEST_INIT_FAILURES:
                    print(
                        "  Test aborted: reached max consecutive sensor init failures "
                        f"after {valid_test_cnt} valid runs and {total_test_cnt} attempts."
                    )
                    break
                continue
            consecutive_init_failures = 0

            test_steps = 0
            test_done = False
            test_imgs = []
            test_success_flag = 0
            prev_action_shoulder = 0.0
            prev_action_arm = 0.0
            test_current_distance = float("inf")
            test_left_any = False
            test_right_any = False

            while not test_done and test_steps < max_steps_per_test_episode:
                test_obs_img, test_obs_tensor = env.get_img(test_steps, test_imgs)
                test_robot_state = env.get_robot_state()
                if len(test_robot_state) < 6:
                    print("  Test warning: robot_state is too short.")
                    break

                safety_features = _build_safety_features(
                    env,
                    test_robot_state,
                    prev_action_shoulder,
                    prev_action_arm,
                    test_steps,
                    max_steps_per_test_episode,
                )
                action_result = gpu.choose_catch(
                    image=test_obs_img,
                    robot_state=test_robot_state,
                    graph_state=test_robot_state,
                    safety_features=safety_features,
                    explore=False,
                    deterministic_eval=True,
                    deterministic_seed=12345 + valid_test_cnt * max_steps_per_test_episode + test_steps,
                    **_catch_eval_schedule()
                )
                action_shoulder_raw = action_result["shoulder_action"]
                action_arm_raw = action_result["arm_action"]

                action_shoulder, action_arm, _, safety_info = _apply_action_safety_filter(
                    env,
                    test_robot_state,
                    _to_scalar(action_shoulder_raw),
                    _to_scalar(action_arm_raw),
                )
                if safety_info["clipped"]:
                    print(
                        f"  Test safety filter: shoulder={action_shoulder:.4f}, "
                        f"arm={action_arm:.4f}, imu_margin={safety_info['imu_margin']:.3f}"
                    )

                _, _, done_from_env, catch_success = _step_catch(
                    env,
                    action_shoulder,
                    action_arm,
                    test_steps,
                )
                test_gps1, _, _, _, _ = env.print_gps()
                if len(test_gps1) >= 3:
                    dy = gps_goal1[0] - test_gps1[1]
                    dz = gps_goal1[1] - test_gps1[2]
                    test_current_distance = (dy * dy + dz * dz) ** 0.5

                _, test_left_any, test_right_any = _get_grasp_contact(env)
                test_success_flag = 1 if catch_success else 0
                prev_action_shoulder = action_shoulder
                prev_action_arm = action_arm
                test_steps += 1

                if done_from_env == 1 or catch_success:
                    test_done = True

            if test_success_flag == 1:
                successful_test_episodes += 1
                print(f"  Test success: L{test_left_any}, R{test_right_any}, distance={test_current_distance:.3f}")
            else:
                print(f"  Test failed: L{test_left_any}, R{test_right_any}, distance={test_current_distance:.3f}")
            valid_test_cnt += 1
    finally:
        gpu.set_mode(stage="catch", mode="train")

    completed_tests = max(1, valid_test_cnt)
    test_success_rate = (successful_test_episodes / completed_tests) * 100.0
    print(
        f"--- Test complete: {successful_test_episodes}/{valid_test_cnt} valid "
        f"(target {NUM_TEST_EPISODES}), success={test_success_rate:.2f}% ---"
    )
    return successful_test_episodes, test_success_rate


def WaveGrad_episoid_1(model_path=None, max_steps_per_episode=22):
    from python_scripts.Webots_interfaces import Environment

    max_steps_per_episode = min(max_steps_per_episode, 22)
    gpu = WaveGradGPUClient(
        host=os.environ.get("WAVEGRAD_GPU_HOST", "127.0.0.1"),
        port=int(os.environ.get("WAVEGRAD_GPU_PORT", "8877")),
    )
    init_info = gpu.initialize(model_path=model_path, max_steps_per_episode=max_steps_per_episode)

    log_writer_catch = Log_write()
    log_writer_tai = Log_write()

    catch_checkpoint_dir = path_list["model_path_catch_WaveGrad"]
    tai_checkpoint_dir = path_list["model_path_tai_WaveGrad"]
    _ensure_dir(catch_checkpoint_dir)
    _ensure_dir(tai_checkpoint_dir)
    _ensure_dir(path_list["catch_log_path_WaveGrad"])
    _ensure_dir(path_list["tai_log_path_WaveGrad"])

    log_file_latest_catch = _next_log_file(path_list["catch_log_path_WaveGrad"], "catch_log")
    log_file_latest_tai = _next_log_file(path_list["tai_log_path_WaveGrad"], "tai_log")
    print(f"Using WaveGrad catch log: {log_file_latest_catch}")
    print(f"Using WaveGrad tai log: {log_file_latest_tai}")

    episode_start = int(init_info.get("episode_start", 0))
    tai_episode = int(init_info.get("tai_episode", 1))
    env = Environment()
    recent_returns = []
    recent_successes = []
    best_test_success_rate = float(init_info.get("best_catch_test_success_rate", -1.0))

    try:
        for episode in range(episode_start, episode_start + 5001):
            print(f"<<<<<<<<< WaveGrad episode {episode}")
            train_schedule = _catch_train_schedule(episode)
            env.reset()
            env.wait(500)
            if not wait_for_sensors_stable(env, max_retries=40, wait_ms=200):
                print("Warning: sensors unstable after reset; attempting environment reset.")
                reset_environment(env)

            steps = 0
            done = 0
            episode_return = 0.0
            success_flag1 = 0
            imgs = []
            prev_distance = None
            prev_action_shoulder = 0.0
            prev_action_arm = 0.0
            episode_safety_penalty = 0.0

            log_writer_catch.add(episode_num=episode)
            while True:
                obs_img, _ = env.get_img(steps, imgs)
                robot_state = env.get_robot_state()
                safety_features = _build_safety_features(
                    env,
                    robot_state,
                    prev_action_shoulder,
                    prev_action_arm,
                    steps,
                    max_steps_per_episode,
                )

                action_result = gpu.choose_catch(
                    image=obs_img,
                    robot_state=robot_state,
                    graph_state=robot_state,
                    safety_features=safety_features,
                    explore=True,
                    **train_schedule
                )
                raw_action_shoulder = _to_scalar(action_result["shoulder_action"])
                raw_action_arm = _to_scalar(action_result["arm_action"])
                value_shoulder = float(action_result["shoulder_value"])
                value_arm = float(action_result["arm_value"])

                action_shoulder, action_arm, safety_penalty, safety_info = _apply_action_safety_filter(
                    env,
                    robot_state,
                    raw_action_shoulder,
                    raw_action_arm,
                )
                if episode % 5 == 0:
                    print(f"Episode {episode}, step {steps}: shoulder={action_shoulder:.4f}, arm={action_arm:.4f}")
                if safety_info["clipped"]:
                    print(
                        "Safety filter: "
                        f"raw=({raw_action_shoulder:.4f}, {raw_action_arm:.4f}) -> "
                        f"safe=({action_shoulder:.4f}, {action_arm:.4f}), "
                        f"penalty={safety_penalty:.3f}, imu_margin={safety_info['imu_margin']:.3f}"
                    )

                gps1, _, _, _, _ = env.print_gps()
                next_state, env_reward, done, catch_success = _step_catch(
                    env,
                    action_shoulder,
                    action_arm,
                    steps,
                )

                gps1, _, _, _, _ = env.print_gps()
                if len(gps1) < 3:
                    dy, dz = 0.0, 0.0
                else:
                    dy = gps_goal1[0] - gps1[1]
                    dz = gps_goal1[1] - gps1[2]
                current_distance = (dy * dy + dz * dz) ** 0.5

                if prev_distance is not None:
                    distance_improvement = prev_distance - current_distance
                    reward = distance_improvement * (3.0 if distance_improvement > 0 else 1.0)
                    reward += max(0.0, (1.0 - current_distance / 2.0)) * 0.5
                else:
                    reward = -current_distance * 0.3
                prev_distance = current_distance

                _, left_any, right_any = _get_grasp_contact(env)
                sensor_triggered = left_any or right_any
                success_flag1 = 1 if catch_success else 0
                if sensor_triggered:
                    if success_flag1 == 1:
                        reward += 50.0
                    else:
                        reward -= 5.0
                if done == 1 and success_flag1 != 1:
                    reward -= 2.0 if steps < 6 else 3.0

                reward += float(env_reward) * 0.1
                reward -= steps * 0.05
                reward -= safety_penalty
                episode_safety_penalty += safety_penalty

                next_obs_img, _ = env.get_img(steps + 1, imgs)
                gpu.store_catch(
                    state=[obs_img, robot_state, robot_state],
                    actions=[action_shoulder, action_arm],
                    reward=reward,
                    next_state=[next_obs_img, next_state, next_state],
                    done=done,
                    values=[value_shoulder, value_arm],
                    safety_features=safety_features,
                    success_flag=success_flag1,
                    safety_penalty=safety_penalty,
                )

                log_writer_catch.add_action_catch(action_shoulder, action_arm)
                episode_return += float(reward)
                prev_action_shoulder = action_shoulder
                prev_action_arm = action_arm
                steps += 1
                if done == 1 or steps >= max_steps_per_episode:
                    break

            learn_info = gpu.learn_catch()
            loss_total = float(learn_info.get("loss", 0.0))
            diffusion_loss = float(learn_info.get("diffusion_loss", 0.0))
            value_loss = float(learn_info.get("value_loss", 0.0))
            policy_lr = float(learn_info.get("policy_lr", 0.0))

            recent_returns.append(episode_return)
            recent_successes.append(success_flag1)
            recent_returns = recent_returns[-100:]
            recent_successes = recent_successes[-100:]
            log_writer_catch.add(loss=loss_total)
            log_writer_catch.add(diffusion_loss=diffusion_loss)
            log_writer_catch.add(value_loss=value_loss)
            log_writer_catch.add(success_replay_size=int(learn_info.get("success_replay_size", 0)))
            log_writer_catch.add(elite_replay_size=int(learn_info.get("elite_replay_size", 0)))
            log_writer_catch.add(policy_lr=policy_lr)
            log_writer_catch.add(return_all=episode_return)
            log_writer_catch.add(goal=success_flag1)
            log_writer_catch.add(safety_penalty=round(episode_safety_penalty, 4))
            if (episode + 1) % 100 == 0:
                rolling_success_rate = 100.0 * sum(recent_successes) / max(1, len(recent_successes))
                rolling_mean_return = sum(recent_returns) / max(1, len(recent_returns))
                log_writer_catch.add(rolling_success_rate_100=rolling_success_rate)
                log_writer_catch.add(rolling_mean_return_100=rolling_mean_return)
                print(
                    f"Rolling 100 episodes: success={rolling_success_rate:.2f}%, "
                    f"mean_return={rolling_mean_return:.2f}"
                )
            print(f"Episode return: {episode_return:.2f}, success: {success_flag1}, loss: {loss_total:.4f}")

            if episode > 0 and episode % CHECKPOINT_INTERVAL == 0:
                successful_test_episodes, test_success_rate = _run_wavegrad_catch_tests(
                    env=env,
                    gpu=gpu,
                    checkpoint_episode=episode,
                    max_steps_per_test_episode=max_steps_per_episode,
                )
                log_writer_catch.add(test_episode=episode)
                log_writer_catch.add(test_grasp_success_rate=test_success_rate)
                log_writer_catch.add(test_score=test_success_rate)
                save_best = test_success_rate > best_test_success_rate
                if save_best:
                    best_test_success_rate = test_success_rate
                    log_writer_catch.add(best_test_grasp_success_rate=best_test_success_rate)
                save_result = gpu.save_catch_checkpoint(
                    episode=episode,
                    episode_return=episode_return,
                    success_flag=success_flag1,
                    test_success_rate=test_success_rate,
                    successful_test_episodes=successful_test_episodes,
                    num_test_episodes=NUM_TEST_EPISODES,
                    save_best=save_best,
                )
                print(f"Saved catch checkpoint through GPU service: {save_result}")
            else:
                print(f"\n--- Episode {episode}: checkpoint test skipped ---")

            log_writer_catch.clear()
            log_writer_catch.save_catch(log_file_latest_catch)

            if success_flag1 == 1:
                print("Catch succeeded; starting WaveGrad tai training.")
                WaveGrad_tai_episoid(
                    gpu=gpu,
                    existing_env=env,
                    total_episode=episode,
                    episode=tai_episode,
                    log_writer_tai=log_writer_tai,
                    log_file_latest_tai=log_file_latest_tai,
                )
                tai_episode += 1
                env.reset()
                env.wait(500)
    finally:
        gpu.close()

    return False, env
