import torch
from python_scripts.SAC.SAC_SACnet import SAC
from python_scripts.SAC.SAC_SACnet_2 import SAC2
from python_scripts.SAC.Replay_memory import ReplayMemory
from python_scripts.SAC.Replay_memory_2 import ReplayMemory_2
from python_scripts.SAC.SAC_episoid_2 import SAC_tai_episoid
from python_scripts.Webots_interfaces import Environment
# from Data_fusion import data_fusion
from python_scripts.Project_config import path_list, gps_goal, gps_goal1, device
from python_scripts.SAC.SAC_Log_write import SAC_Log_write
from python_scripts.utils.sensor_utils import wait_for_sensors_stable, reset_environment
import numpy as np
import heapq
import os


class ModelRanking:
    """
    追踪和管理前 N 个最佳 SAC 模型的辅助类：
    - 与 PPO 版本逻辑保持一致，使用最小堆维护排行榜
    - 新模型优于当前最差模型时，自动删除最差模型文件
    """
    def __init__(self, top_n=5, key_name='success_rate'):
        self.top_n = top_n
        self.rankings = []
        self.key_name = key_name
        self.saved_paths = []

    def add_and_manage(self, new_score, new_checkpoint, episode_id, base_dir):
        """
        根据新模型的评分（成功率）决定是否保存，并在需要时删除旧模型。
        """
        new_entry = (new_score, "")

        should_save = False
        final_save_path = ""

        if len(self.rankings) < self.top_n:
            should_save = True
            final_save_path = os.path.join(base_dir, f'sac_model_success_{episode_id}.ckpt')
        elif new_score > self.rankings[0][0]:
            should_save = True
            final_save_path = os.path.join(base_dir, f'sac_model_success_{episode_id}.ckpt')
            worst_score, worst_path_to_delete = heapq.heappop(self.rankings)
            try:
                os.remove(worst_path_to_delete)
                print(f"删除旧 SAC 模型文件: {worst_path_to_delete} (成功率: {worst_score:.2f}%)")
            except FileNotFoundError:
                print(f"警告: 试图删除不存在的文件 {worst_path_to_delete}")

        if should_save:
            torch.save(new_checkpoint, final_save_path)
            new_entry = (new_score, final_save_path)
            heapq.heappush(self.rankings, new_entry)
            print(f"SAC 模型 {episode_id} (成功率: {new_score:.2f}%) 已保存到 {final_save_path} 并加入排行榜。")
            return final_save_path
        else:
            print(f"SAC 模型 {episode_id} (成功率: {new_score:.2f}%) 未进入前 {self.top_n}，未保存。")
            return None

    def print_current_rankings(self):
        """打印当前排行榜内容。"""
        if not self.rankings:
            print("当前 SAC 排行榜为空。")
            return

        print("\n--- 基于测试成功率的最佳 SAC 模型排行榜 ---")
        sorted_rankings = sorted(self.rankings, key=lambda x: x[0], reverse=True)
        for i, (score, path) in enumerate(sorted_rankings, 1):
            ep_num = path.split('_')[-1].split('.')[0]
            print(f"  {i}. Episode {ep_num}: Success Rate = {score:.2f}%, Path = {path}")
        print("-----------------------------------------\n")


def SAC_episoid(model_path=None):
    # 创建SAC算法对象，将act_dim从2改为连续动作空间的维度
    sac = SAC(act_dim=2, node_num=19)
    sac2 = SAC2()

    # 初始化日志写入器
    log_writer_catch = SAC_Log_write()  # 创建抓取日志写入器
    log_writer_tai = SAC_Log_write()  # 创建抬腿日志写入器
    import os
    import glob
    import re
    # 查找现有的日志文件，确定最新的编号
    # 抓取阶段：
    log_pattern = os.path.join(path_list['catch_log_path_SAC'], 'catch_log_*.json')
    existing_logs = glob.glob(log_pattern)
    latest_num = 0
    if existing_logs:
        # 从文件名中提取编号
        for log_path in existing_logs:
            match = re.search(r'catch_log_(\d+)', log_path)
            if match:
                num = int(match.group(1))
                latest_num = max(latest_num, num)
        # 新的日志文件编号
        new_log_num = latest_num + 1
    else:
        # 没有现有日志文件，从1开始
        new_log_num = 1
    log_file_latest_catch = os.path.join(path_list['catch_log_path_SAC'], f"catch_log_{new_log_num}.json")
    print(f"将使用新的抓取日志目录: {log_file_latest_catch}")

    # 抬腿阶段：
    log_pattern = os.path.join(path_list['tai_log_path_SAC'], 'tai_log_*.json')
    existing_logs = glob.glob(log_pattern)
    latest_num = 0
    if existing_logs:
        # 从文件名中提取编号
        for log_path in existing_logs:
            match = re.search(r'tai_log_(\d+)', log_path)
            if match:
                num = int(match.group(1))
                latest_num = max(latest_num, num)
        # 新的日志文件编号
        new_log_num = latest_num + 1
    else:
        # 没有现有日志文件，从1开始
        new_log_num = 1
    log_file_latest_tai = os.path.join(path_list['tai_log_path_SAC'], f"tai_log_{new_log_num}.json")
    print(f"将使用新抬腿的日志目录: {log_file_latest_tai}")

    # 加载模型
    # 抓取模型加载
    if model_path:  # 如果指定了模型路径
        try:
            # 从指定路径加载模型
            checkpoint = torch.load(model_path)
            sac.policy_net.load_state_dict(checkpoint['policy_net'])
            sac.q_net.load_state_dict(checkpoint['q_net'])
            sac.target_q_net.load_state_dict(checkpoint['target_q_net'])
            sac.log_alpha = checkpoint['log_alpha']
            sac.alpha = torch.exp(sac.log_alpha)
            # 从文件名中提取周期数
            episode_start = int(model_path.split('_')[-1].split('.')[0])
            print(f"从指定模型加载: {model_path}，从周期 {episode_start} 继续训练")
            print("模型加载成功！")
        except Exception as e:
            print(f"指定模型加载失败: {e}")
            episode_start = 0
    else:  # 如果没有指定模型路径，使用原来的自动查找逻辑
        # 获取所有模型文件
        model_files = glob.glob(path_list['model_path_catch_SAC'] + '/sac_model_*.ckpt')
        if model_files:
            # 按文件名中的数字排序，获取最新的模型文件
            latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            episode_start = int(latest_model.split('_')[-1].split('.')[0])
            print(f"找到最新抓取模型: {latest_model}，从周期 {episode_start} 继续训练")
            
            # 加载模型
            try:
                checkpoint = torch.load(latest_model)
                sac.policy_net.load_state_dict(checkpoint['policy_net'])
                sac.q_net.load_state_dict(checkpoint['q_net'])
                sac.target_q_net.load_state_dict(checkpoint['target_q_net'])
                sac.log_alpha = checkpoint['log_alpha']
                sac.alpha = torch.exp(sac.log_alpha)
                print("抓取模型加载成功！")
            except Exception as e:
                print(f"抓取模型加载失败: {e}")
                episode_start = 0
        else:
            print("未找到已保存的抓取模型，从头开始训练")
            episode_start = 0
    
    # 抬腿模型加载
    model_files_tai = glob.glob(path_list['model_path_tai_SAC'] + '/sac_model_tai_*.ckpt')
    if model_files_tai:
        try:
            # 按新的文件名格式排序：dqn_model_tai_{total_episoid}_{episode}.ckpt
            # 定义一个函数来提取total_episoid和episode
            def extract_numbers(filename):
                # 从文件名中提取数字部分
                parts = filename.split('_')
                if len(parts) >= 5:  # 确保文件名格式正确
                    try:
                        total_ep = int(parts[-2])  # 倒数第二个是total_episoid
                        ep = int(parts[-1].split('.')[0])  # 最后一个是episode（去掉.ckpt）
                        return (total_ep, ep)
                    except (ValueError, IndexError):
                        return (0, 0)  # 解析失败时返回默认值
                return (0, 0)
            
            # 按照total_episoid和episode排序，找出最新的模型
            latest_model = max(model_files_tai, key=extract_numbers)
            total_ep, ep = extract_numbers(latest_model)
            print(f"找到最新抬腿模型: {latest_model}，总周期: {total_ep}，抬腿周期: {ep}")
            
            checkpoint = torch.load(latest_model)
            sac2.policy_net.load_state_dict(checkpoint['policy_net'])
            sac2.q_net.load_state_dict(checkpoint['q_net'])
            sac2.target_q_net.load_state_dict(checkpoint['target_q_net'])
            sac2.log_alpha = checkpoint['log_alpha']
            sac2.alpha = torch.exp(sac2.log_alpha)
            print("抬腿模型加载成功！")
        except Exception as e:
            print(f"抬腿模型加载失败: {e}")
    else:
        print("未找到已保存的抬腿模型，从头开始训练")

    tai_episoid = 1
    episode_num = episode_start  # 初始化回合计数器
    rpm = ReplayMemory(30000)  # 创建经验回放缓存（缩小为 30000，更偏向近期数据）
    rpm_2 = ReplayMemory_2(100000)
    env = Environment()

    # SAC算法的训练更新次数
    SAC_UPDATES_PER_STEP = 1

    # 模型评估 & 排行榜参数（与 PPO 逻辑保持一致）
    top_n_models = 5
    model_ranking = ModelRanking(top_n=top_n_models)
    CHECKPOINT_INTERVAL = 500   # 每 500 周期做一次评估
    NUM_TEST_EPISODES = 100     # 每次评估统计 100 个有效 episode
    MAX_TEST_ATTEMPTS = 5       # 初始化失败的最大重试次数

    for i in range(episode_start, episode_start + 50000):  # 从episode_start开始，最多再训练50000个周期
        log_writer_catch.add(episode_num=i)
        print(f"<<<<<<<<<第{i}周期") # 打印当前周期
        env.reset()
        env.wait(500)   # 等待500ms
        # 使用与 PPO 相同的方式检查传感器稳定性，避免无效数据
        if not wait_for_sensors_stable(env, max_retries=40, wait_ms=200):
            print("警告: 传感器不稳定，尝试重置环境...")
            reset_environment(env)
        imgs = []  # 初始化图像列表
        steps = 0  # 初始化步数
        return_all = 0  # 初始化总奖励
        prev_distance = None
        # 动作平滑相关变量（与 PPO 保持一致，减少动作抖动）
        prev_action_shoulder = 0.0
        prev_action_arm = 0.0
        action_smooth_alpha = 0.7  # 平滑系数
        obs_img, obs_tensor = env.get_img(steps, imgs)  # 获取初始图像和图像张量
        # log_writer_catch.add(obs_img=obs_img, steps=steps)
        robot_state = env.get_robot_state()  # 获取机器人状态
        # print(f'robot_state: {robot_state}')
        # print(f'robot_state_len: {len(robot_state)}')
        print("____________________")  # 打印初始状态
        while True:
            # print(f'第{episode_num}周期，第{steps}步')
            sac_state = [robot_state[1], robot_state[0], robot_state[5], robot_state[4]]  # 将机器人状态转换为SAC状态向量
            obs = [obs_img, sac_state]
            # log_writer_catch.add(obs=obs, steps=steps)
            # 输入次数、状态，选择动作
            # SAC 输出连续动作（2 维：肩膀/手臂），与 PPO 的动作空间保持一致
            continuous_action = sac.choose_action(
                episode_num=episode_num,
                obs=obs,
                x_graph=robot_state
            )

            # 解析连续动作为肩膀和手臂两个关节
            if isinstance(continuous_action, np.ndarray):
                action_shoulder = float(continuous_action[0])
                action_arm = float(continuous_action[1]) if continuous_action.shape[0] > 1 else float(continuous_action[0])
            else:
                # 兼容性处理：如果只返回一个标量，则两个关节共用
                action_shoulder = float(continuous_action)
                action_arm = float(continuous_action)

            # 动作平滑（指数移动平均），减少抖动，和 PPO 保持一致
            action_shoulder_smooth = action_smooth_alpha * prev_action_shoulder + (1 - action_smooth_alpha) * action_shoulder
            action_arm_smooth = action_smooth_alpha * prev_action_arm + (1 - action_smooth_alpha) * action_arm
            prev_action_shoulder = action_shoulder_smooth
            prev_action_arm = action_arm_smooth

            print(f'第{i}周期，第{steps}步，肩膀动作(原始/平滑): {action_shoulder:.4f}/{action_shoulder_smooth:.4f}，手臂动作(原始/平滑): {action_arm:.4f}/{action_arm_smooth:.4f}')
            
            # env.wait(1000)
            # print('wait 1000ms')
            
            gps1, gps2, gps3, gps4, foot_gps1 = env.print_gps()  # 获取GPS位置
            if steps >= 19:  # 如果步数大于等于19
                catch_flag = 1.0  # 抓取器状态为1.0
            else:
                catch_flag = 0.0  # 抓取器状态为0.0
            img_name = "img" + str(steps) + ".png"  # 图像名称
            # 添加动作到日志（保留原有接口）
            log_writer_catch.add_continuous_action(continuous_action)  # 添加连续动作记录

            # 执行一步动作（使用平滑后的连续动作，与 PPO 一致）
            next_state, reward_env, done, good, goal, count = env.step(
                robot_state,
                action_shoulder_smooth,
                action_arm_smooth,
                steps,
                catch_flag,
                gps1, gps2, gps3, gps4,
                img_name
            )
            print(f'catch_flag: {catch_flag}')
            print(f'done: {done}')
            print(f'【调试】环境返回: reward_env={reward_env:.2f}, goal={goal}, good={good}')

            # === 与 PPO 对齐的距离奖励设计 ===
            gps1, _, _, _, _ = env.print_gps()
            if len(gps1) < 3:
                print(f"警告：gps1长度不足 ({len(gps1)} < 3)，使用默认值")
                dx = 0.0
                dy = 0.0
            else:
                dx = gps_goal[0] - gps1[1]
                dy = gps_goal[1] - gps1[2]
            current_distance = (dx ** 2 + dy ** 2) ** 0.5

            # 距离变化奖励（鼓励靠近目标）
            if prev_distance is not None:
                reward = (prev_distance - current_distance) * 15.0
            else:
                reward = -current_distance
            prev_distance = current_distance

            # 抓取传感器组合判断，与 PPO 保持一致
            all_grasp_sensors = [
                env.darwin.get_touch_sensor_value('grasp_L1'),
                env.darwin.get_touch_sensor_value('grasp_L1_1'),
                env.darwin.get_touch_sensor_value('grasp_L1_2'),
                env.darwin.get_touch_sensor_value('grasp_R1'),
                env.darwin.get_touch_sensor_value('grasp_R1_1'),
                env.darwin.get_touch_sensor_value('grasp_R1_2')
            ]
            left_sensors = all_grasp_sensors[0:3]
            right_sensors = all_grasp_sensors[3:6]
            left_any = any(left_sensors)
            right_any = any(right_sensors)
            success_flag1 = 1 if (left_any and right_any) else 0
            if env.is_collision():
                reward -= 50

            if success_flag1 == 1:
                if current_distance <= 0.04:
                    reward += 300
                    print("✅ 抓到目标梯级，发放大奖励！")
                else:
                    reward -= 160
                    print("⚠️ 抓到非目标梯级，无大奖励")

            if done == 1 and steps < 6 and success_flag1 != 1:
                print("错误抓取！给予较大惩罚！")
                reward -= 100
            if done == 1 and steps >= 6 and success_flag1 != 1:
                print("错误抓取！给予较大惩罚！")
                reward -= 100
            if done == 1 and steps <= 2 and success_flag1 != 1:
                print("因环境不稳定导致无效数据，跳过此步骤！！！")
                break

            # 步长惩罚
            reward -= 10

            return_all = return_all + reward  # 总奖励为当前奖励加上之前的总奖励
            steps += 1  # 步数加1

            # 添加奖励和步数记录（记录我们重新计算的 reward）
            log_writer_catch.add_reward(reward)
            log_writer_catch.add_return(return_all)
            log_writer_catch.add_step(steps)
            log_writer_catch.add_goal(goal)
            
            next_obs_img, next_obs_tensor = env.get_img(steps, imgs)  # 获取下一个图像和图像张量
            next_obs = [next_obs_img, next_state]
            # print('获取下一个状态更新完毕')
            # 只跳过环境明显不稳定的前几步样本，其余与 PPO 一样全部存储
            should_store = True
            if done == 1 and steps <= 2 and success_flag1 != 1:
                should_store = False
                print(f"  跳过无效样本：done={done}, steps={steps}, success={success_flag1}")

            if should_store:
                # 将当前状态、动作、奖励、下一个状态、是否完成、是否达到目标添加到经验回放缓存中
                # 为SAC准备连续动作空间（使用连续动作，而不是离散 a）
                rpm.append((obs_img, robot_state, continuous_action, reward, next_obs_img, next_state, done))
            robot_state = env.get_robot_state()  # 获取机器人状态
            obs_tensor = next_obs_tensor  # 更新图像张量
            if len(rpm) < 5000:  # 如果经验回放缓存小于5000
                episode_num = 0  # 计数器为0
            if len(rpm) > 5000 and done == 1:  # 只有在buffer中存满了数据才会学习
                # if goal == 1:  # 如果达到目标，额外保存一份普通 checkpoint
                #     print("goal = 1")
                #     save_path = path_list['model_path_catch_SAC'] + '/sac_model_%s.ckpt' % i
                #     checkpoint = {
                #         'policy_net': sac.policy_net.state_dict(),
                #         'q_net': sac.q_net.state_dict(),
                #         'target_q_net': sac.target_q_net.state_dict(),
                #         'log_alpha': sac.log_alpha
                #     }
                #     torch.save(checkpoint, save_path)
                    
                # SAC学习，进行多次更新
                q_loss_sum = 0
                policy_loss_sum = 0
                alpha_loss_sum = 0
                
                for _ in range(SAC_UPDATES_PER_STEP):
                    q_loss, policy_loss, alpha_loss = sac.learn(rpm)
                    q_loss_sum += q_loss
                    policy_loss_sum += policy_loss
                    alpha_loss_sum += alpha_loss
                
                # 计算平均损失
                avg_q_loss = q_loss_sum / SAC_UPDATES_PER_STEP
                avg_policy_loss = policy_loss_sum / SAC_UPDATES_PER_STEP
                avg_alpha_loss = alpha_loss_sum / SAC_UPDATES_PER_STEP
                
                # 记录损失值和温度参数
                log_writer_catch.add_losses(avg_q_loss, avg_policy_loss, avg_alpha_loss)
                log_writer_catch.add_alpha(sac.log_alpha, sac.alpha)
                
                print(f"Q损失: {avg_q_loss}, 策略损失: {avg_policy_loss}, Alpha损失: {avg_alpha_loss}")
                
                # 每500步保存一次模型（普通 checkpoint）
                # if i % 500 == 0:
                #     path = path_list['model_path_catch_SAC'] + '/sac_model_%s.ckpt' % i
                #     checkpoint = {
                #         'policy_net': sac.policy_net.state_dict(),
                #         'q_net': sac.q_net.state_dict(),
                #         'target_q_net': sac.target_q_net.state_dict(),
                #         'log_alpha': sac.log_alpha
                #     }
                #     torch.save(checkpoint, path)
                #     print(f"保存模型: {path}")
                
                # 写入总奖励
                log_writer_catch.add_return(return_all)
                # 写入目标
                log_writer_catch.add_goal(goal)

                # --- 准备通用 checkpoint 数据，用于测试评估和排行榜 ---
                base_checkpoint_data = {
                    'policy_net': sac.policy_net.state_dict(),
                    'q_net': sac.q_net.state_dict(),
                    'target_q_net': sac.target_q_net.state_dict(),
                    'log_alpha': sac.log_alpha,
                    'episode': i
                }

                # --- 检查是否到达评估检查点 ---
                is_checkpoint_interval = (i % CHECKPOINT_INTERVAL == 0) and (i != 0)
                if is_checkpoint_interval:
                    print(f"\n--- SAC 周期 {i}: 到达检查点，开始在当前环境进行模型测试 (共 {NUM_TEST_EPISODES} 轮有效) ---")
                    sac.policy_net.eval()

                    successful_test_episodes = 0
                    valid_test_cnt = 0           # 已经跑完的有效轮次
                    total_test_cnt = 0           # 总共开启的轮次（含无效）
                    max_steps_per_test_episode = 500

                    while valid_test_cnt < NUM_TEST_EPISODES:
                        total_test_cnt += 1
                        print(f"———— SAC 测试轮次 {valid_test_cnt + 1}/{NUM_TEST_EPISODES} (总开启 {total_test_cnt}) ————")

                        # 1. 初始化，确保传感器稳定
                        is_test_valid = False
                        for init_try in range(MAX_TEST_ATTEMPTS):
                            env.reset()
                            env.wait(200)
                            if wait_for_sensors_stable(env, max_retries=40, wait_ms=200):
                                is_test_valid = True
                                break
                            print(f"  警告: 传感器不稳定，尝试重置... ({init_try + 1}/{MAX_TEST_ATTEMPTS})")

                        if not is_test_valid:
                            print("  ❌ 初始化失败，此轮不计入有效统计。")
                            continue

                        # 2. 跑一个测试 episode（纯评估模式，关闭探索）
                        test_steps, test_done = 0, False
                        test_imgs = []
                        while not test_done and test_steps < max_steps_per_test_episode:
                            test_obs_img, test_obs_tensor = env.get_img(test_steps, test_imgs)
                            test_robot_state = env.get_robot_state()
                            if len(test_robot_state) < 6:
                                print("  测试警告：robot_state 长度不足，提前结束本轮。")
                                break

                            test_sac_state = [test_robot_state[1], test_robot_state[0], test_robot_state[5], test_robot_state[4]]
                            test_obs = [test_obs_img, test_sac_state]

                            with torch.no_grad():
                                continuous_action_t = sac.choose_action(
                                    episode_num=i,
                                    obs=test_obs,
                                    x_graph=test_robot_state,
                                    evaluate=True
                                )

                            if isinstance(continuous_action_t, np.ndarray):
                                action_shoulder_t = float(continuous_action_t[0])
                                action_arm_t = float(continuous_action_t[1]) if continuous_action_t.shape[0] > 1 else float(continuous_action_t[0])
                            else:
                                action_shoulder_t = float(continuous_action_t)
                                action_arm_t = float(continuous_action_t)

                            action_shoulder_t = np.clip(action_shoulder_t, -0.5, 0.5)
                            action_arm_t = np.clip(action_arm_t, -0.5, 0.5)

                            test_gps1, test_gps2, test_gps3, test_gps4, _ = env.print_gps()
                            if len(test_gps1) < 3:
                                test_steps += 1
                                continue

                            test_catch_flag = 1.0 if test_steps >= 19 else 0.0
                            _, _, test_done_from_env, _, test_goal_from_env, _ = env.step(
                                test_robot_state,
                                action_shoulder_t,
                                action_arm_t,
                                test_steps,
                                test_catch_flag,
                                test_gps1, test_gps2, test_gps3, test_gps4,
                                f"sac_test_img_{test_steps}.png"
                            )
                            if test_done_from_env or test_goal_from_env:
                                test_done = True
                            test_steps += 1

                        # 3. 判定结果（与 PPO 判定逻辑保持一致）
                        final_touch = env.darwin.get_touch_sensor_value('grasp_L1_2')
                        early_fail = (test_steps <= 2 and final_touch != 1)
                        gps1_final, _, _, _, _ = env.print_gps()
                        if len(gps1_final) < 3:
                            print(f"警告：gps1长度不足 ({len(gps1_final)} < 3)，使用默认值")
                            dx = 0.0
                            dy = 0.0
                        else:
                            dx = gps_goal[0] - gps1_final[1]
                            dy = gps_goal[1] - gps1_final[2]
                        current_distance_final = (dx ** 2 + dy ** 2) ** 0.5

                        if early_fail:
                            print("  ❌ 过早结束且未成功，此轮无效。")
                            continue
                        elif (final_touch == 1 or test_goal_from_env) and current_distance_final <= 0.04:
                            successful_test_episodes += 1
                            print("  ✓ 测试成功！")
                        else:
                            print("  ✗ 测试失败。")

                        valid_test_cnt += 1

                    sac.policy_net.train()

                    test_success_rate = successful_test_episodes
                    log_writer_catch.add(success_rate=test_success_rate)
                    print(f"\n--- SAC 测试完成：{NUM_TEST_EPISODES} 轮测试成功率为 {test_success_rate:.2f}% ---")

                    model_ranking.add_and_manage(
                        new_score=test_success_rate,
                        new_checkpoint=base_checkpoint_data,
                        episode_id=i,
                        base_dir=path_list['model_path_catch_SAC']
                    )
                    model_ranking.print_current_rankings()
                
            success_flag1 = env.darwin.get_touch_sensor_value('grasp_L1_2')

            if catch_flag == 1.0 or done == 1:  # 如果抓取器状态为1.0或完成
                # 写入重置标志
                # if(success_flag1 == 0):
                #     env.reset()  # 重置环境
                env.wait(100)  # 等待100ms
                imgs = []  # 初始化图像列表
                steps = 0  # 初始化步数
                episode_num = episode_num + 1  # 计数器加1
                # obs, obs_tensor = env.get_img(steps, imgs)  # 获取初始图像和图像张量
                # robot_state = env.get_robot_state()  # 获取机器人状态
                #log_writer_catch.add(action_list=log_writer_catch.action_list)
                log_writer_catch.clear()  # 清除当前序列，准备记录新的序列
                log_writer_catch.save_catch(log_file_latest_catch)  # 保存日志
                break
        
        if success_flag1 == 1:
            print("抓取成功，开始抬腿训练...")
            total_episoid = i
            print("tai_episoid:", tai_episoid)
            SAC_tai_episoid(sac2=sac2, existing_env=env, total_episoid=total_episoid, episode=tai_episoid, rpm_2=rpm_2, log_writer_tai=log_writer_tai, log_file_latest_tai=log_file_latest_tai)
            tai_episoid += 1


    # 如果整个训练过程结束，返回抓取成功状态和环境实例
    return False, env