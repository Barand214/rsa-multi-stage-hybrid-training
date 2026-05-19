import math

from python_scripts.WaveGrad.WaveGrad_log_write import Log_write
from python_scripts.Project_config import gps_goal1, path_list


def _tai_train_schedule(episode):
    if episode < 300:
        return {
            "explore_noise_std": 0.06,
            "q_guidance_probability": 0.0,
            "candidate_count": 4,
            "action_clip": 0.85,
        }
    if episode < 1000:
        return {
            "explore_noise_std": 0.04,
            "q_guidance_probability": 0.5,
            "candidate_count": 4,
            "action_clip": 0.85,
        }
    if episode < 2000:
        return {
            "explore_noise_std": 0.02,
            "q_guidance_probability": 0.8,
            "candidate_count": 4,
            "action_clip": 0.85,
        }
    return {
        "explore_noise_std": 0.005,
        "q_guidance_probability": 0.95,
        "candidate_count": 4,
        "action_clip": 0.85,
    }


def WaveGrad_tai_episoid(
    gpu=None,
    wavegrad_tai_leg_upper=None,
    wavegrad_tai_leg_lower=None,
    wavegrad_tai_ankle=None,
    existing_env=None,
    total_episode=0,
    episode=0,
    log_writer_tai=None,
    log_file_latest_tai=None,
):
    from python_scripts.Webots_interfaces import Environment

    trajectory_len = 20
    owns_gpu = gpu is None
    if gpu is None:
        from python_scripts.WaveGrad.wavegrad_gpu_client import WaveGradGPUClient

        gpu = WaveGradGPUClient()
        gpu.initialize(max_steps_per_episode=trajectory_len)
    if log_writer_tai is None:
        log_writer_tai = Log_write()

    env = existing_env if existing_env is not None else Environment()
    print("Starting WaveGrad tai stage.")
    env.darwin.tai_leg_L1()
    env.darwin.tai_leg_L2()

    return_all = 0.0
    imgs = []
    goal = 0
    done = 0
    steps = 0
    catch_flag = 0.0
    train_schedule = _tai_train_schedule(episode)

    log_writer_tai.add(episode_num=total_episode)
    while True:
        obs_img, _ = env.get_img(steps, imgs)
        robot_state = env.get_robot_state()
        action_result = gpu.choose_tai(
            image=obs_img,
            robot_state=robot_state,
            graph_state=robot_state,
            explore=True,
            **train_schedule
        )
        action_leg_upper = float(action_result["leg_upper_action"])
        action_leg_lower = float(action_result["leg_lower_action"])
        action_ankle = float(action_result["ankle_action"])
        value_leg_upper = float(action_result["leg_upper_value"])
        value_leg_lower = float(action_result["leg_lower_value"])
        value_ankle = float(action_result["ankle_value"])

        log_writer_tai.add_action_tai(action_leg_upper, action_leg_lower, action_ankle)
        if episode % 5 == 0:
            print(f"Tai step {steps + 1}: {action_leg_upper:.4f}, {action_leg_lower:.4f}, {action_ankle:.4f}")

        gps_values = env.print_gps()
        next_state, reward, done, good, goal, count = env.step2(
            robot_state,
            action_leg_upper,
            action_leg_lower,
            action_ankle,
            steps,
            catch_flag,
            gps_values[4],
            gps_values[0],
            gps_values[1],
            gps_values[2],
            gps_values[3],
        )

        if count == 1:
            x1 = gps_goal1[0] - gps_values[4][1]
            y1 = gps_goal1[1] - gps_values[4][2]
            distance = math.sqrt(x1 * x1 + y1 * y1)
            if distance > 0.06:
                reward = 0.0
            elif distance > 0.03:
                reward = 0.1
            else:
                reward = 1.0

        return_all += float(reward)
        steps += 1
        next_obs_img, _ = env.get_img(steps, imgs)

        if good == 1:
            success_flag = bool(goal)
            gpu.store_tai(
                state=[obs_img, robot_state, robot_state],
                actions=[action_leg_upper, action_leg_lower, action_ankle],
                reward=reward,
                next_state=[next_obs_img, robot_state, next_state],
                done=done,
                values=[value_leg_upper, value_leg_lower, value_ankle],
                success_flag=success_flag,
            )

        if steps > trajectory_len:
            done = 1

        if episode % 400 == 0 and done == 1:
            save_result = gpu.save_tai_checkpoint(total_episode=total_episode, tai_episode=episode)
            print(f"Saving WaveGrad tai checkpoint: {save_result}")

        if episode > 0 and done == 1:
            learn_info = gpu.learn_tai()
            log_writer_tai.add(loss=float(learn_info.get("loss", 0.0)))
            log_writer_tai.add(diffusion_loss=float(learn_info.get("diffusion_loss", 0.0)))
            log_writer_tai.add(value_loss=float(learn_info.get("value_loss", 0.0)))
            log_writer_tai.add(q_loss=float(learn_info.get("q_loss", 0.0)))
            q_guidance_loss = float(learn_info.get("q_guidance_loss", 0.0))
            log_writer_tai.add(q_guidance_loss=q_guidance_loss)
            log_writer_tai.add(q_guidance_loss_used=float(learn_info.get("q_guidance_loss_used", q_guidance_loss)))
            log_writer_tai.add(q_guidance_loss_ratio=float(learn_info.get("q_guidance_loss_ratio", 0.0)))
            log_writer_tai.add(success_replay_size=int(learn_info.get("success_replay_size", 0)))
            log_writer_tai.add(elite_replay_size=int(learn_info.get("elite_replay_size", 0)))
            log_writer_tai.add(q_guided_used=int(learn_info.get("q_guided_used", 0)))
            log_writer_tai.add(q_guided_action_delta=float(learn_info.get("q_guided_action_delta", 0.0)))
            log_writer_tai.add(critic_updates=int(learn_info.get("critic_updates", 0)))
            log_writer_tai.add(return_all=return_all)
            log_writer_tai.add(goal=goal)

        if done == 1 or steps > trajectory_len:
            print("WaveGrad tai stage finished; resetting robot.")
            env.darwin._set_left_leg_initpose()
            env.darwin.robot_reset()
            for _ in range(40):
                env.robot.step(env.timestep)
            env.wait(1000)
            if log_file_latest_tai:
                log_writer_tai.save_tai(log_file_latest_tai)
            if owns_gpu:
                gpu.close()
            break
