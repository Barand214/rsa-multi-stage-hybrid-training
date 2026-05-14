# 傅里叶版本的训练代码 - 基于PPO_episoid_1.py修改
import torch
import shutil
import heapq
import os
import glob
import re
import numpy as np
from python_scripts.PPO.PPO_PPOnet import PPO
from python_scripts.Webots_interfaces import Environment
from python_scripts.Project_config import path_list, gps_goal, gps_goal1, device, Darwin_config
from python_scripts.PPO_Log_write import Log_write
from python_scripts.utils.sensor_utils import wait_for_sensors_stable, reset_environment

class ModelRanking:
    """
    一个用于追踪和管理N个最佳模型的辅助类。
    它使用最小堆来高效地找到当前性能最差的模型。
    """
    def __init__(self, top_n=5, key_name='success_rate'): 
        self.top_n = top_n
        self.rankings = []
        self.key_name = key_name
        self.saved_paths = []

    def add_and_manage(self, new_score, new_checkpoint, episode_id, base_dir):
        """
        核心方法：根据新模型的分数和排行榜情况，决定是否保存模型文件。
        """
        new_entry = (new_score, "") 

        should_save = False
        final_save_path = ""

        # 如果排行榜未满，直接保存
        if len(self.rankings) < self.top_n:
            should_save = True
            final_save_path = os.path.join(base_dir, f'ppo_model_fourier_{episode_id}.ckpt')
        # 如果排行榜已满，但新模型比最差的要好
        elif new_score > self.rankings[0][0]:
            should_save = True
            final_save_path = os.path.join(base_dir, f'ppo_model_fourier_{episode_id}.ckpt')
            worst_score, worst_path_to_delete = heapq.heappop(self.rankings)
            try:
                os.remove(worst_path_to_delete)
                print(f"删除旧模型文件: {worst_path_to_delete} (成功率: {worst_score:.2f}%)")
            except FileNotFoundError:
                print(f"警告: 试图删除不存在的文件 {worst_path_to_delete}")

        if should_save:
            torch.save(new_checkpoint, final_save_path)
            new_entry = (new_score, final_save_path)
            heapq.heappush(self.rankings, new_entry)
            print(f"模型 {episode_id} (成功率: {new_score:.2f}%) 已保存到 {final_save_path} 并加入排行榜。")
            return final_save_path
        else:
            print(f"模型 {episode_id} (成功率: {new_score:.2f}%) 性能未进入前 {self.top_n}，未保存。")
            return None

    def print_current_rankings(self):
        """打印当前排行榜内容。"""
        if not self.rankings:
            print("当前排行榜为空。")
            return
            
        print("\n--- 基于测试成功率的最佳模型排行榜（傅里叶版本）---")
        sorted_rankings = sorted(self.rankings, key=lambda x: x[0], reverse=True)
        for i, (score, path) in enumerate(sorted_rankings, 1):
            ep_num = path.split('_')[-1].split('.')[0]
            print(f"  {i}. Episode {ep_num}: Success Rate = {score:.2f}%, Path = {path}")
        print("-----------------------------------------\n")


def PPO_episoid_fourier(model_path=None, max_steps_per_episode=500, n_servos=2, max_harmonics=1, T_max=2.0, step_interval=0.1):   
    """
    傅里叶版本的PPO训练函数
    
    参数:
        model_path: 预训练模型路径
        max_steps_per_episode: 每个episode的最大步数
        n_servos: 舵机数量
        max_harmonics: 最大谐波数
        T_max: 最大周期时间
        step_interval: 时间步长间隔
    """
    # 创建傅里叶版本的PPO对象
    ppo_fourier = PPO(
        node_num=19, 
        env_information=None,
        act_dim=n_servos,  # 傅里叶模式下act_dim表示舵机数量
        use_fourier=True,
        n_servos=n_servos,
        max_harmonics=max_harmonics,
        T_max=T_max,
        step_interval=step_interval
    )
    
    # 初始化排行榜
    top_n_models = 5
    model_ranking = ModelRanking(top_n=top_n_models)
    
    # 初始化日志写入器
    log_writer = Log_write()
    
    # ✅ 修改：每次学习后都保存和测试，不再使用固定间隔
    # CHECKPOINT_INTERVAL = 500  # 已废弃：改为每次学习后都执行
    NUM_TEST_EPISODES = 100  
    MAX_TEST_ATTEMPTS = 5 
    
    # 查找现有的日志文件，确定最新的编号
    log_pattern = os.path.join(path_list['catch_log_path_PPO'], 'fourier_log_*.json')
    existing_logs = glob.glob(log_pattern)
    latest_num = 0
    if existing_logs:
        for log_path in existing_logs:
            match = re.search(r'fourier_log_(\d+)', log_path)
            if match:
                num = int(match.group(1))
                latest_num = max(latest_num, num)
        new_log_num = latest_num + 1
    else:
        new_log_num = 1
    log_file_latest = os.path.join(path_list['catch_log_path_PPO'], f"fourier_log_{new_log_num}.json")
    print(f"将使用新的傅里叶日志目录: {log_file_latest}")

    # 加载模型
    if model_path:
        try:
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict) and 'policy' in checkpoint:
                ppo_fourier.policy.load_state_dict(checkpoint['policy'], strict=False)
                if 'optimizer' in checkpoint and ppo_fourier.optimizer:
                    ppo_fourier.optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"从指定模型加载: {model_path}，模型加载成功！")
                episode_start = int(model_path.split('_')[-1].split('.')[0])
                print(f"从指定模型加载: {model_path}，从周期 {episode_start} 继续训练")
            else:
                ppo_fourier.policy.load_state_dict(checkpoint, strict=False)
                print(f"指定模型文件 {model_path} 格式不匹配或不是字典格式，从头开始训练。")
                episode_start = 0
        except Exception as e:
            print(f"指定模型加载失败: {e}")
            episode_start = 0
    else:
        # ✅ 修改：自动查找性能最好的模型（而非最新的模型）
        model_files = glob.glob(path_list['model_path_catch_PPO'] + '/ppo_model_fourier_*.ckpt')
        if model_files:
            print(f"\n找到 {len(model_files)} 个傅里叶模型文件，正在扫描性能数据...")
            
            best_model = None
            best_score = -1
            best_episode = 0
            
            # 遍历所有模型文件，找出性能最好的
            for model_file in model_files:
                try:
                    checkpoint = torch.load(model_file, map_location=device)
                    if isinstance(checkpoint, dict) and 'success_rate' in checkpoint:
                        score = checkpoint['success_rate']
                        episode = checkpoint.get('episode', int(model_file.split('_')[-1].split('.')[0]))
                        print(f"  - {os.path.basename(model_file)}: Episode {episode}, 成功率 {score:.2f}%")
                        
                        if score > best_score:
                            best_score = score
                            best_model = model_file
                            best_episode = episode
                    else:
                        # 旧格式模型，没有成功率信息
                        episode = int(model_file.split('_')[-1].split('.')[0])
                        print(f"  - {os.path.basename(model_file)}: Episode {episode}, 无性能数据（旧格式）")
                except Exception as e:
                    print(f"  - {os.path.basename(model_file)}: 读取失败 ({e})")
            
            # 如果找到了有性能数据的模型，加载最好的
            if best_model:
                latest_model = best_model
                episode_start = best_episode
                print(f"\n✅ 选择性能最佳模型: {os.path.basename(latest_model)}")
                print(f"   Episode: {episode_start}, 成功率: {best_score:.2f}%")
            else:
                # 如果所有模型都是旧格式，回退到按 episode 编号选择最新的
                latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                episode_start = int(latest_model.split('_')[-1].split('.')[0])
                print(f"\n⚠️  所有模型均无性能数据，回退到加载最新模型: {os.path.basename(latest_model)}")
                print(f"   Episode: {episode_start}")
            
            try:
                checkpoint = torch.load(latest_model)
                if isinstance(checkpoint, dict) and 'policy' in checkpoint:
                    ppo_fourier.policy.load_state_dict(checkpoint['policy'], strict=False)
                    if 'optimizer' in checkpoint and ppo_fourier.optimizer:
                        ppo_fourier.optimizer.load_state_dict(checkpoint['optimizer'])
                    print("傅里叶模型加载成功！")
                else:
                    ppo_fourier.policy.load_state_dict(checkpoint, strict=False)
                    print("傅里叶模型加载成功！(旧格式)")
            except Exception as e:
                print(f"傅里叶模型加载失败: {e}")
                episode_start = 0
        else:
            print("未找到已保存的傅里叶模型，从头开始训练")
            episode_start = 0

    episode_num = episode_start
    env = Environment()
    success_count = 0
    learn_counter = 0
    LEARN_INTERVAL = 20  # 每10个成功episode学习一次（只从成功经验中学习）

    for i in range(episode_start, episode_start + 30000):
        log_writer.add(episode_num=i)
        print(f"<<<<<<<<<第{i}周期（傅里叶版本）")
        success_flag = 0
        env.reset()
        env.wait(500)
        
        # 检查传感器状态
        if not wait_for_sensors_stable(env, max_retries=40, wait_ms=200):
            print("警告: 传感器不稳定，尝试重置环境...")
            reset_environment(env)
        
        imgs = []
        steps = 0
        return_all = 0
        obs_img, obs_tensor = env.get_img(steps, imgs)
        robot_state = env.get_robot_state()
        print("____________________")
        prev_distance = None
        
        # === 【傅里叶模式】生成整个周期的动作参数 ===
        ppo_state = [robot_state[1], robot_state[0], robot_state[5], robot_state[4]]
        obs = (obs_tensor, robot_state)
        
        # 选择傅里叶动作参数
        action_params, log_prob, value, _ = ppo_fourier.choose_action(
            episode_num=i, 
            obs=obs, 
            x_graph=robot_state
        )
        
        print(f"傅里叶参数: n={action_params['n']}, T={action_params['T']:.3f}")
        print(f"  A范围: [{action_params['A'].min():.3f}, {action_params['A'].max():.3f}]")
        print(f"  ω范围: [{action_params['ω'].min():.3f}, {action_params['ω'].max():.3f}]")
        print(f"  φ范围: [{action_params['φ'].min():.3f}, {action_params['φ'].max():.3f}]")
        
        # 【关键】将傅里叶参数转换为numpy数组，避免重复计算
        action_params_np = {
            'n': action_params['n'],
            'T': action_params['T'],
            'A': action_params['A'].cpu().numpy() if torch.is_tensor(action_params['A']) else action_params['A'],
            'ω': action_params['ω'].cpu().numpy() if torch.is_tensor(action_params['ω']) else action_params['ω'],
            'φ': action_params['φ'].cpu().numpy() if torch.is_tensor(action_params['φ']) else action_params['φ']
        }
        
        # 【调试】打印第一个舵机的傅里叶函数，验证是否会产生变化
        if i % 50 == 0:  # 每50个episode打印一次
            print(f"【调试】傅里叶函数测试 (舵机0):")
            for t_test in [0.0, 0.05, 0.1, 0.15, 0.2]:
                angle_test = ppo_fourier.action_space.get_fourier_curve(action_params_np, t_test)
                print(f"  t={t_test:.2f}s: 肩膀={angle_test[0]:.4f}, 手臂={angle_test[1]:.4f}")
        
        # === 【关键】执行傅里叶动作序列 ===
        current_time = 0.0
        episode_done = False
        episode_rewards = []  # 记录本episode的所有奖励
        episode_stored_count = 0  # 记录本episode存储的样本数
        
        while not episode_done and steps < max_steps_per_episode:
            # 安全检查
            if len(robot_state) < 6:
                print(f"警告：robot_state长度不足 ({len(robot_state)} < 6)，跳过此步")
                break
            
            # 根据当前时间计算舵机角度（时间在周期T内循环）
            time_in_cycle = current_time % action_params_np['T']
            angles = ppo_fourier.action_space.get_fourier_curve(action_params_np, time_in_cycle)
            
            # 提取肩膀和手臂动作（假设前2个舵机）
            action_shoulder = float(angles[0])
            action_arm = float(angles[1])
            
            # 【调试】每个episode的前3步详细打印，帮助诊断
            if steps < 3:
                print(f'【详细调试】第{i}周期，第{steps}步:')
                print(f'  current_time={current_time:.3f}s, T={action_params_np["T"]:.3f}s, time_in_cycle={time_in_cycle:.3f}s')
                print(f'  angles数组: {angles}')
                print(f'  肩膀={action_shoulder:.4f}, 手臂={action_arm:.4f}')
            elif steps % 10 == 0:  # 每10步打印一次，减少输出
                print(f'第{i}周期，第{steps}步，时间={current_time:.3f}s(周期内={time_in_cycle:.3f}s), 肩膀={action_shoulder:.4f}, 手臂={action_arm:.4f}')
            
            gps1, gps2, gps3, gps4, foot_gps1 = env.print_gps()
            catch_flag = 1.0 if steps >= 19 else 0.0
            img_name = f"fourier_img_{steps}.png"
            
            # 执行动作
            next_state, reward_env, done, good, goal, count = env.step(
                robot_state, action_shoulder, action_arm, steps, 
                catch_flag, gps1, gps2, gps3, gps4, img_name
            )
            
            if steps % 10 == 0:  # 每10步打印一次
                print(f'catch_flag: {catch_flag}, done: {done}')
                print(f'【调试】环境返回: reward_env={reward_env:.2f}, goal={goal}, good={good}')
            
            # === 计算奖励（与原版相同的奖励逻辑）===
            gps1, _, _, _, _ = env.print_gps()
            if len(gps1) < 3:
                print(f"警告：gps1长度不足 ({len(gps1)} < 3)，使用默认值")
                dx = 0.0
                dy = 0.0
            else:
                dx = gps_goal[0] - gps1[1]
                dy = gps_goal[1] - gps1[2]
            current_distance = (dx**2 + dy**2)**0.5

            # 距离变化奖励
            success_flag = env.darwin.get_touch_sensor_value('grasp_L1_2')
            if prev_distance is not None:
                reward = (prev_distance - current_distance) * 5.0
            else:
                reward = -current_distance

            prev_distance = current_distance

            # 抓取判断
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
            success_flag = 1 if (left_any and right_any) else 0
            
            if success_flag == 1:
                if current_distance <= 0.04:
                    reward += 300
                    print("✅ 抓到目标梯级，发放大奖励！")
                else:
                    reward -= 160
                    print("⚠️  抓到非目标梯级，无大奖励")
            
            if done == 1 and steps < 6 and success_flag != 1:
                print("错误抓取！给予较大惩罚！")
                reward -= 100
            if done == 1 and steps >= 6 and success_flag != 1:
                print("错误抓取！给予较大惩罚！")
                reward -= 100
            if done == 1 and steps <= 2 and success_flag != 1:
                print("因环境不稳定导致无效数据，跳过此步骤！！！")
                episode_done = True
                break
            
            return_all = return_all + reward
            
            steps += 1
            current_time += step_interval  # 时间步进
            next_obs_img, next_obs_tensor = env.get_img(steps, imgs)
            next_obs = [next_obs_img, next_state]
            
            # ✅ 新逻辑：直接存储所有数据，不再暂存
            STORE_INTERVAL = 5  # 每5步存储一次
            should_store = (steps % STORE_INTERVAL == 0) or done == 1 or success_flag == 1
            
            if should_store:
                # === 方案B：为每条transition计算其对应 state 下的 old_log_prob(s_t,a) 与 V(s_t) ===
                # 注意：这里的 action_params 是 episode 级固定参数，但 log_prob/value 必须按当前 state 重新评估
                step_log_prob, step_value = ppo_fourier.evaluate_fourier_log_prob_value(
                    obs=(obs_tensor, robot_state),
                    x_graph=robot_state,
                    action_params=action_params
                )
                step_next_value = ppo_fourier.evaluate_value(
                    obs=(next_obs_tensor, next_state),
                    x_graph=next_state
                )
                
                # ✅ 直接存储到真正的缓冲区
                ppo_fourier.store_transition_catch(
                    state=[obs_tensor, robot_state, robot_state],
                    action=action_params,
                    reward=reward,
                    next_state=[next_obs_tensor, next_state, next_state],
                    done=done,
                    value=step_value,
                    next_value=step_next_value,
                    log_prob=step_log_prob,
                    image=obs_img,
                    angles=angles,
                    time_step=time_in_cycle
                )
                episode_stored_count += 1
            
            # ❌ 旧逻辑：暂存到临时缓冲区（已注释）
            # if should_store:
            #     # 暂存到episode级别的缓冲区，等episode结束后再决定是否真正存储
            #     if not hasattr(ppo_fourier, 'temp_episode_buffer'):
            #         ppo_fourier.temp_episode_buffer = []
            #
            #     # === 方案B：为每条transition计算其对应 state 下的 old_log_prob(s_t,a) 与 V(s_t) ===
            #     # 注意：这里的 action_params 是 episode 级固定参数，但 log_prob/value 必须按当前 state 重新评估
            #     step_log_prob, step_value = ppo_fourier.evaluate_fourier_log_prob_value(
            #         obs=(obs_tensor, robot_state),
            #         x_graph=robot_state,
            #         action_params=action_params
            #     )
            #     step_next_value = ppo_fourier.evaluate_value(
            #         obs=(next_obs_tensor, next_state),
            #         x_graph=next_state
            #     )
            #     
            #     ppo_fourier.temp_episode_buffer.append({
            #         # 【修复1】训练与采样保持一致：使用 obs_tensor / next_obs_tensor 作为图像输入
            #         'state': [obs_tensor, robot_state, robot_state],
            #         'action': action_params,
            #         'reward': reward,
            #         'next_state': [next_obs_tensor, next_state, next_state],
            #         'done': done,
            #         'value': step_value,
            #         'next_value': step_next_value,
            #         'log_prob': step_log_prob,
            #         'image': obs_img,
            #         'angles': angles,
            #         'time_step': time_in_cycle
            #     })
            #     episode_stored_count += 1
            
            robot_state = env.get_robot_state()
            obs_tensor = next_obs_tensor
            obs_img = next_obs_img
            
            # 检查是否结束
            if done == 1 or steps >= max_steps_per_episode:
                episode_done = True
                
                # 判断本episode是否成功
                episode_success = (success_flag == 1 and current_distance <= 0.04)
                
                # ✅ 新逻辑：所有数据都已经直接存储，这里只做统计
                if episode_success:
                    print(f"\n✅ Episode {i} 成功！数据已存储（共 {episode_stored_count} 个样本）")
                else:
                    print(f"\n⚠️  Episode {i} 失败，但数据仍然存储（共 {episode_stored_count} 个样本，用于学习）")
                
                # ❌ 旧逻辑：只存储成功的episode（已注释）
                # if episode_success:
                #     print(f"\n✅ Episode {i} 成功！将数据存储到学习缓冲区")
                #     if hasattr(ppo_fourier, 'temp_episode_buffer'):
                #         for transition in ppo_fourier.temp_episode_buffer:
                #             ppo_fourier.store_transition_catch(
                #                 state=transition['state'],
                #                 action=transition['action'],
                #                 reward=transition['reward'],
                #                 next_state=transition['next_state'],
                #                 done=transition['done'],
                #                 value=transition['value'],
                #                 next_value=transition.get('next_value', None),
                #                 log_prob=transition['log_prob'],
                #                 image=transition['image'],
                #                 angles=transition['angles'],
                #                 time_step=transition['time_step']
                #             )
                #         print(f"  已存储 {len(ppo_fourier.temp_episode_buffer)} 个样本到缓冲区")
                # else:
                #     print(f"\n❌ Episode {i} 失败，丢弃本episode的数据")
                # 
                # # 清空临时缓冲区
                # ppo_fourier.temp_episode_buffer = []
                
                # 打印本episode统计信息
                print(f"\n=== Episode {i} 结束 ===")
                print(f"  总步数: {steps}")
                print(f"  暂存样本数: {episode_stored_count}")
                print(f"  累积奖励: {return_all:.2f}")
                print(f"  平均奖励: {return_all/steps:.2f}")
                print(f"  成功标志: {success_flag}")
                if len(episode_rewards) > 0:
                    print(f"  奖励范围: [{min(episode_rewards):.2f}, {max(episode_rewards):.2f}]")
                
                # ✅ 新逻辑：每个episode都计数，不区分成功失败
                learn_counter += 1
                print(f"📊 Episode计数: {learn_counter}/{LEARN_INTERVAL} (成功={episode_success})")
                
                # 标记是否进行了学习（用于后续决定是否测试和保存）
                did_learn_this_episode = False
                
                if learn_counter >= LEARN_INTERVAL:
                    print(f"\n--- 已累积 {learn_counter} 个episode，开始学习 ---")
                    print(f"  当前缓冲区大小: {len(ppo_fourier.rewards)} 个样本")
                    if len(ppo_fourier.rewards) == 0:
                        print(f"  ⚠️ 警告：没有数据可学习，跳过本次学习")
                        loss = 0.0
                    else:
                        loss = ppo_fourier.learn()
                        print(f'  学习完成，loss: {loss:.4f}')
                        did_learn_this_episode = True  # 标记本次进行了学习
                    log_writer.add(loss=loss)
                    learn_counter = 0
                else:
                    print(f"--- Episode累积中... ({learn_counter}/{LEARN_INTERVAL}) ---")
                    loss = 0.0
                    log_writer.add(loss=loss)
                
                # ❌ 旧逻辑：只在成功的episode后累加计数器（已注释）
                # if episode_success:
                #     learn_counter += 1
                #     print(f"✅ 成功episode计数: {learn_counter}/{LEARN_INTERVAL}")
                #     
                #     if learn_counter >= LEARN_INTERVAL:
                #         print(f"\n--- 已累积 {learn_counter} 个成功episode，开始学习 ---")
                #         print(f"  当前缓冲区大小: {len(ppo_fourier.rewards)} 个样本")
                #         if len(ppo_fourier.rewards) == 0:
                #             print(f"  ⚠️ 警告：没有数据可学习，跳过本次学习")
                #             loss = 0.0
                #         else:
                #             loss = ppo_fourier.learn()
                #             print(f'  学习完成，loss: {loss:.4f}')
                #         log_writer.add(loss=loss)
                #         learn_counter = 0
                #     else:
                #         print(f"--- 成功episode累积中... ({learn_counter}/{LEARN_INTERVAL}) ---")
                #         loss = 0.0
                #         log_writer.add(loss=loss)
                # else:
                #     print(f"❌ 失败episode，不计入学习计数器（当前: {learn_counter}/{LEARN_INTERVAL}）")
                #     loss = 0.0
                #     log_writer.add(loss=loss)

                # 准备checkpoint数据（暂时不包含成功率，等测试后再添加）
                base_checkpoint_data = {
                    'policy': ppo_fourier.policy.state_dict(),
                    'optimizer': ppo_fourier.optimizer.state_dict(),
                    'episode': i,
                    'fourier_params': {
                        'n_servos': n_servos,
                        'max_harmonics': max_harmonics,
                        'T_max': T_max,
                        'step_interval': step_interval
                    }
                }

                # ✅ 修改：每次学习后都进行模型评估和保存
                # 旧逻辑：is_checkpoint_interval = (i % CHECKPOINT_INTERVAL == 0) and (i != 0)
                if did_learn_this_episode:
                    print(f"\n--- 周期 {i}: 到达检查点，开始模型测试 (共 {NUM_TEST_EPISODES} 轮) ---")
                    ppo_fourier.policy.eval()

                    successful_test_episodes = 0
                    valid_test_cnt = 0
                    total_test_cnt = 0
                    max_steps_per_test_episode = 500

                    while valid_test_cnt < NUM_TEST_EPISODES:
                        total_test_cnt += 1
                        print(f"————————————————测试轮次 {valid_test_cnt + 1}/{NUM_TEST_EPISODES} "
                              f"(总开启 {total_test_cnt})——————————————")

                        # 初始化
                        is_test_valid = False
                        for init_try in range(MAX_TEST_ATTEMPTS):
                            env.reset()
                            env.wait(200)
                            if wait_for_sensors_stable(env, max_retries=40, wait_ms=200):
                                is_test_valid = True
                                break
                            print(f"  警告: 传感器不稳定，尝试重置... ({init_try + 1}/{MAX_TEST_ATTEMPTS})")

                        if not is_test_valid:
                            print(f"  ❌ 初始化失败，此轮不计入有效统计。")
                            continue

                        # 运行测试episode
                        test_steps, test_done = 0, False
                        test_imgs = []
                        test_time = 0.0
                        
                        # 生成测试用的傅里叶参数
                        test_obs_img, test_obs_tensor = env.get_img(test_steps, test_imgs)
                        test_robot_state = env.get_robot_state()
                        test_obs = (test_obs_tensor, test_robot_state)
                        
                        with torch.no_grad():
                            test_action_params, _, _, _ = ppo_fourier.choose_action(
                                episode_num=i, obs=test_obs, x_graph=test_robot_state, explore=False)
                        
                        while not test_done and test_steps < max_steps_per_test_episode:
                            test_robot_state = env.get_robot_state()
                            if len(test_robot_state) < 6:
                                print("  测试警告：robot_state 长度不足，提前结束本轮。")
                                break

                            # 计算当前时间的舵机角度
                            test_angles = ppo_fourier.action_space.get_fourier_curve(test_action_params, test_time)
                            action_shoulder_t = float(test_angles[0])
                            action_arm_t = float(test_angles[1])
                            
                            action_shoulder_t = np.clip(action_shoulder_t, -0.5, 0.5)
                            action_arm_t = np.clip(action_arm_t, -0.5, 0.5)

                            test_gps1, test_gps2, test_gps3, test_gps4, _ = env.print_gps()
                            if len(test_gps1) < 3:
                                test_steps += 1
                                test_time += step_interval
                                continue

                            test_catch_flag = 1.0 if test_steps >= 19 else 0.0
                            _, _, test_done_from_env, _, test_goal_from_env, _ = env.step(
                                test_robot_state, action_shoulder_t, action_arm_t, test_steps,
                                test_catch_flag, test_gps1, test_gps2, test_gps3, test_gps4,
                                f"test_fourier_img_{test_steps}.png")
                            
                            if test_done_from_env or test_goal_from_env:
                                test_done = True
                            test_steps += 1
                            test_time += step_interval

                        # 判定结果
                        final_touch = env.darwin.get_touch_sensor_value('grasp_L1_2')
                        early_fail = (test_steps <= 2 and final_touch != 1)
                        gps1, _, _, _, _ = env.print_gps()
                        
                        if len(gps1) < 3:
                            print(f"警告：gps1长度不足 ({len(gps1)} < 3)，使用默认值")
                            dx = 0.0
                            dy = 0.0
                        else:
                            dx = gps_goal[0] - gps1[1]
                            dy = gps_goal[1] - gps1[2]
                        current_distance = (dx**2 + dy**2)**0.5
                        
                        if early_fail:
                            print(f"  ❌ 过早结束且未成功，此轮无效。")
                            continue
                        elif (final_touch == 1 or test_goal_from_env) and current_distance <= 0.04:
                            successful_test_episodes += 1
                            print(f"  ✓ 测试成功！")
                        else:
                            print(f"  ✗ 测试失败。")

                        valid_test_cnt += 1

                    ppo_fourier.policy.train()
                    test_success_rate = successful_test_episodes
                    log_writer.add(success_rate=test_success_rate)
                    print(f"\n--- 测试完成：{NUM_TEST_EPISODES}轮测试成功率为 {test_success_rate:.2f}% ---")
                    
                    # ✅ 将成功率添加到 checkpoint 中
                    base_checkpoint_data['success_rate'] = test_success_rate
                    
                    # 更新排行榜
                    model_ranking.add_and_manage(
                        new_score=test_success_rate,
                        new_checkpoint=base_checkpoint_data,
                        episode_id=i,
                        base_dir=path_list['model_path_catch_PPO']
                    )

                    model_ranking.print_current_rankings()

                else:
                    print(f"\n--- 周期 {i}: 未进行学习，跳过模型测试和保存 ---")

                # 记录日志
                print(f"本轮训练累积奖励: {return_all:.2f}, 目标达成: {success_flag}")
                log_writer.add(log_prob=log_prob)
                log_writer.add(value=value)
                log_writer.add(return_all=return_all)
                log_writer.add(goal=1 if success_flag else 0)
                log_writer.clear()
                log_writer.save_catch(log_file_latest)
                
                break

        # Episode结束处理
        if catch_flag == 1.0 or done == 1:
            env.wait(100)
            imgs = []
            steps = 0
            episode_num = episode_num + 1
            log_writer.clear()
            log_writer.save_catch(log_file_latest)

        if success_flag == 1:
            success_count += 1
            log_writer.add(success_catch=success_count)
            print("success_count:", success_count)
            print("抓取成功！")

    log_writer.save_catch(log_file_latest)
    return False, env


if __name__ == "__main__":
    # 运行傅里叶版本的训练
    PPO_episoid_fourier(
        model_path=None,
        max_steps_per_episode=500,
        n_servos=20,
        max_harmonics=20,
        T_max=10.0,
        step_interval=0.1
    )

