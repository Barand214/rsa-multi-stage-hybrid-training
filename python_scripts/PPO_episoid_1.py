# 测试
import torch
import shutil
import heapq
import os
import glob
import re
import numpy as np
from python_scripts.PPO.PPO_PPOnet_2 import PPO2
from python_scripts.PPO.PPO_PPOnet import PPO
from python_scripts.PPO.Replay_memory import ReplayMemory
from python_scripts.PPO.Replay_memory_2 import ReplayMemory_2
from python_scripts.PPO.PPO_episoid_2_1 import PPO_tai_episoid
from python_scripts.Webots_interfaces import Environment
# from Data_fusion import data_fusion
from python_scripts.Project_config import path_list, gps_goal, gps_goal1, device, Darwin_config
from python_scripts.PPO_Log_write import Log_write
from python_scripts.PPO.RobotRun1 import RobotRun 
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
            final_save_path = os.path.join(base_dir, f'ppo_model_success_{episode_id}.ckpt')
        # 如果排行榜已满，但新模型比最差的要好
        elif new_score > self.rankings[0][0]:
            should_save = True
            final_save_path = os.path.join(base_dir, f'ppo_model_success_{episode_id}.ckpt')
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
            
        print("\n--- 基于测试成功率的最佳模型排行榜 ---")
        sorted_rankings = sorted(self.rankings, key=lambda x: x[0], reverse=True)
        for i, (score, path) in enumerate(sorted_rankings, 1):
            ep_num = path.split('_')[-1].split('.')[0]
            # --- 修改：打印信息改为成功率 ---
            print(f"  {i}. Episode {ep_num}: Success Rate = {score:.2f}%, Path = {path}")
        print("-----------------------------------------\n")


def PPO_episoid_1(model_path=None, max_steps_per_episode=500):   
    #ppo_catch = PPO(node_num=19, env_information=None, act_dim=2)  # 创建一个合并后的PPO对象
    ppo_arm = PPO(node_num=19, env_information=None,act_dim=1)  # 创建PPO对象
    ppo_shoulder = PPO(node_num=19, env_information=None,act_dim=1)  # 创建PPO对象

    ppo2_LegUpper = PPO2(node_num=19, env_information=None)  # 创建PPO2对象
    ppo2_LegLower = PPO2(node_num=19, env_information=None)  # 创建PPO2对象
    ppo2_Ankle = PPO2(node_num=19, env_information=None)  # 创建PPO2对象
    # --- 初始化排行榜 ---
    top_n_models = 5
    model_ranking = ModelRanking(top_n=top_n_models)
    # 初始化日志写入器
    log_writer_catch = Log_write()  # 创建抓取日志写入器
    log_writer_tai = Log_write()  # 创建抬腿日志写入器
    CHECKPOINT_INTERVAL = 500  
    NUM_TEST_EPISODES = 100  
    MAX_TEST_ATTEMPTS = 5 
    tai_episoid = 1
    
    # 查找现有的日志文件，确定最新的编号
    # 抓取阶段：
    log_pattern = os.path.join(path_list['catch_log_path_PPO'], 'catch_log_*.json')
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
    log_file_latest_catch = os.path.join(path_list['catch_log_path_PPO'], f"catch_log_{new_log_num}.json")
    print(f"将使用新的抓取日志目录: {log_file_latest_catch}")

    # 抬腿阶段：
    log_pattern = os.path.join(path_list['tai_log_path_PPO'], 'tai_log_*.json')
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
    log_file_latest_tai = os.path.join(path_list['tai_log_path_PPO'], f"tai_log_{new_log_num}.json")
    print(f"将使用新抬腿的日志目录: {log_file_latest_tai}")

    # 加载模型
    # 抓取模型加载
    if model_path:  # 如果指定了模型路径
        try:
            # 从指定路径加载模型
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict) and 'policy_shoulder' in checkpoint:
                # 如果是保存的字典格式 {'policy': state_dict, ...}
                # 增加 strict=False 以兼容添加 LMF 后新增的参数键
                ppo_shoulder.policy.load_state_dict(checkpoint['policy_shoulder'], strict=False)
                ppo_arm.policy.load_state_dict(checkpoint['policy_arm'], strict=False)
                # 如果需要加载优化器状态
                if 'optimizer_shoulder' in checkpoint and ppo_shoulder.optimizer:
                    ppo_shoulder.optimizer.load_state_dict(checkpoint['optimizer_shoulder'])
                if 'optimizer_arm' in checkpoint and ppo_arm.optimizer:
                    ppo_arm.optimizer.load_state_dict(checkpoint['optimizer_arm'])
                print(f"从指定模型加载: {model_path}，模型加载成功！")
                episode_start = int(model_path.split('_')[-1].split('.')[0])
                print(f"从指定模型加载: {model_path}，从周期 {episode_start} 继续训练")
            else:
                # 如果是直接保存的模型或状态字典
                ppo_shoulder.policy.load_state_dict(checkpoint, strict=False)
                ppo_arm.policy.load_state_dict(checkpoint, strict=False)
                print(f"指定模型文件 {model_path} 格式不匹配或不是字典格式，从头开始训练。")
                episode_start = 0
        except Exception as e:
            print(f"指定模型加载失败: {e}")
            episode_start = 0
    else:  # 如果没有指定模型路径，使用原来的自动查找逻辑
        # 获取所有模型文件
        model_files = glob.glob(path_list['model_path_catch_PPO'] + '/ppo_model_*.ckpt')
        if model_files:
            # 按文件名中的数字排序，获取最新的模型文件
            latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            episode_start = int(latest_model.split('_')[-1].split('.')[0])
            print(f"找到最新抓取模型: {latest_model}，从周期 {episode_start} 继续训练")
            
            # 加载模型
            try:
                checkpoint = torch.load(latest_model)
                if isinstance(checkpoint, dict) and 'policy_shoulder' in checkpoint:
                    # 如果是保存的字典格式 {'policy': state_dict, ...}
                    ppo_shoulder.policy.load_state_dict(checkpoint['policy_shoulder'], strict=False)
                    ppo_arm.policy.load_state_dict(checkpoint['policy_arm'], strict=False)
                    # 如果需要加载优化器状态
                    if 'optimizer_shoulder' in checkpoint and ppo_shoulder.optimizer:
                        ppo_shoulder.optimizer.load_state_dict(checkpoint['optimizer_shoulder'])
                    if 'optimizer_arm' in checkpoint and ppo_arm.optimizer:
                        ppo_arm.optimizer.load_state_dict(checkpoint['optimizer_arm'])
                    print("抓取模型加载成功！")
                else:
                    # 如果是直接保存的模型或状态字典
                    ppo_shoulder.policy.load_state_dict(checkpoint, strict=False)
                    print("抓取模型加载成功！(旧格式)")
                    
            except Exception as e:
                print(f"抓取模型加载失败: {e}")
                episode_start = 0
        else:
            print("未找到已保存的抓取模型，从头开始训练")
            episode_start = 0
    
    # 抬腿模型加载
    model_files_tai = glob.glob(path_list['model_path_tai_PPO'] + '/ppo_model_tai_*.ckpt')
    if model_files_tai:
        try:
            # 按新的文件名格式排序：ppo_model_tai_{total_episoid}_{episode}.ckpt
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
            tai_episoid = ep
            print(f"抬腿模型从周期 {tai_episoid} 继续训练")
            # 加载模型
            try:
                checkpoint = torch.load(latest_model)
                # 判断是否为新版字典格式（包含各关节 policy 键）
                if isinstance(checkpoint, dict) and 'policy_LegUpper' in checkpoint:
                    # 如果是保存的字典格式 {'policy_LegUpper': state_dict, ...}
                    ppo2_LegUpper.policy.load_state_dict(checkpoint['policy_LegUpper'])
                    ppo2_LegLower.policy.load_state_dict(checkpoint['policy_LegLower'])
                    ppo2_Ankle.policy.load_state_dict(checkpoint['policy_Ankle'])
                    # 如果需要加载优化器状态
                    if 'optimizer_LegUpper' in checkpoint and ppo2_LegUpper.optimizer:
                        ppo2_LegUpper.optimizer.load_state_dict(checkpoint['optimizer_LegUpper'])
                    if 'optimizer_LegLower' in checkpoint and ppo2_LegLower.optimizer:
                        ppo2_LegLower.optimizer.load_state_dict(checkpoint['optimizer_LegLower'])
                    if 'optimizer_Ankle' in checkpoint and ppo2_Ankle.optimizer:
                        ppo2_Ankle.optimizer.load_state_dict(checkpoint['optimizer_Ankle'])
                    print("抬腿模型加载成功！")
                else:
                    # 如果是直接保存的模型或状态字典
                    ppo2_LegUpper.policy.load_state_dict(checkpoint)
                    ppo2_LegLower.policy.load_state_dict(checkpoint)
                    ppo2_Ankle.policy.load_state_dict(checkpoint)
                    print("抬腿模型加载成功！(旧格式)")
            except Exception as e:
                print(f"抬腿模型加载失败: {e}")
        except Exception as e:
            print(f"抬腿模型加载失败: {e}")
    else:
        print("未找到已保存的抬腿模型，从头开始训练")




    episode_num = episode_start  # 初始化回合计数器
    env = Environment()
    success_catch = 0                  # 抓取成功次数
    learn_counter = 0                  # 学习计数器：每10个episode学习一次
    LEARN_INTERVAL = 10                # 每10个episode学习一次

    for i in range(episode_start, episode_start + 10000):  # 从episode_start开始，最多再训练50000个周期
        log_writer_catch.add(episode_num=i)
        print(f"<<<<<<<<<第{i}周期") # 打印当前周期
        success_flag1 = 0
        env.reset()
        env.wait(500)                         
        # 使用工具函数检查传感器状态
        if not wait_for_sensors_stable(env, max_retries=40, wait_ms=200):
            print("警告: 传感器不稳定，尝试重置环境...")
            reset_environment(env)
        imgs = []  # 初始化图像列表
        steps = 0  # 初始化步数
        return_all = 0  # 初始化总奖励
        obs_img, obs_tensor = env.get_img(steps, imgs)  # 获取初始图像和图像张量
        robot_state = env.get_robot_state()  # 获取机器人状态
        print("____________________")  # 打印初始状态
        prev_distance = None
        prev_distance_wrong = None  # 【修复】初始化错误目标距离记录
        # 【优化】初始化动作平滑变量（用于动作平滑机制）
        prev_action_shoulder = 0.0
        prev_action_arm = 0.0
        action_smooth_alpha = 0.7  # 平滑系数：0.7表示保留70%的上一动作，30%的新动作
        while True:
            # 安全检查：确保robot_state有足够的元素
            if len(robot_state) < 6:
                print(f"警告：robot_state长度不足 ({len(robot_state)} < 6)，跳过此步")
                continue
            ppo_state = [robot_state[1], robot_state[0], robot_state[5], robot_state[4]]  # 将机器人状态转换为ppo状态
            # log_writer.add(ppo_state=ppo_state, steps=steps)
            obs = (obs_tensor, robot_state)
            # log_writer.add(obs=obs, steps=steps)
            # 将机器人状态转换为张量
            # x_graph = torch.tensor(robot_state, dtype=torch.float32).to(device)
            # x_graph = torch.tensor(robot_state, dtype=torch.float32).unsqueeze(1).to(device)  # 添加维度
            # 输入次数、状态，选择动作
            # actions_combined, log_prob_combined, value_combined = ppo_catch.choose_action(episode_num=i, obs=obs, x_graph=robot_state, action_type='shoulder') # action_type参数已废弃，可忽略
    
            # # --- 分离出两个动作 ---
            # # 现在的结果是一个2元素的 numpy 数组
            # action_shoulder = actions_combined[0] 
            # action_arm = actions_combined[1]
            # ppo_shoulder 的 act_dim=1，所以它会返回一个标量动作
            action_shoulder, log_prob_shoulder, value_shoulder, action_raw_shoulder = ppo_shoulder.choose_action(episode_num=i, obs=obs, x_graph=robot_state)
            action_arm, log_prob_arm, value_arm, action_raw_arm = ppo_arm.choose_action(episode_num=i, obs=obs, x_graph=robot_state)

            # 现在它们的输出就是真正的标量，可以直接处理，不需要 [0] 索引了！
            # 但为了应对万一返回的仍然是 [1] 的形状，再加一层保护总是好的
            s_val = float(action_shoulder[0]) if hasattr(action_shoulder, 'shape') and action_shoulder.shape else float(action_shoulder)
            a_val = float(action_arm[0]) if hasattr(action_arm, 'shape') and action_arm.shape else float(action_arm)

            # 【优化】动作平滑：使用指数移动平均来平滑动作，减少动作抖动
            # 公式：smooth_action = alpha * prev_action + (1 - alpha) * new_action
            action_shoulder_smooth = action_smooth_alpha * prev_action_shoulder + (1 - action_smooth_alpha) * s_val
            action_arm_smooth = action_smooth_alpha * prev_action_arm + (1 - action_smooth_alpha) * a_val
            
            # 更新上一动作（用于下一次平滑）
            prev_action_shoulder = action_shoulder_smooth
            prev_action_arm = action_arm_smooth

            print(f'第{i}周期，第{steps}步，肩膀动作(原始/平滑): {s_val:.4f}/{action_shoulder_smooth:.4f}，手臂动作(原始/平滑): {a_val:.4f}/{action_arm_smooth:.4f}')
            
            gps1, gps2, gps3, gps4, foot_gps1 = env.print_gps()  # 获取GPS位置
            if steps >= 19:  # 如果步数大于等于19
                catch_flag = 1.0  # 抓取器状态为1.0
            else:
                catch_flag = 0.0  # 抓取器状态为0.0
            img_name = "img" + str(steps) + ".png"  # 图像名称
            # print("action:", a)
            # 分别添加动作、对数概率和状态价值到日志
            log_writer_catch.add_action_catch(action_shoulder, action_arm) 
            #log_writer_catch.add_log_prob_catch(log_prob_combined, log_prob_combined) # 使用合并后的 log_prob
            #log_writer_catch.add_value_catch(value_combined, value_combined)
            # 执行一步动作（使用平滑后的动作）
            next_state, reward_env, done, good, goal, count = env.step(robot_state, action_shoulder_smooth, action_arm_smooth, steps, catch_flag, gps1, gps2, gps3, gps4, img_name)
            # next_state, reward, done, good, goal, count = RobotRun(
            #     env.darwin, robot_state, action_shoulder, action_arm, steps, 
            #     catch_flag, gps1, gps2, gps3, gps4, img_name
            # ).run()
            
            # === 【关键修复】忽略环境返回的reward，完全使用我们计算的reward ===
            # 问题：RobotRun1.py 中 goal=1 时会设置 reward=100，导致网络可能学习"达到goal"而不是"最大化reward"
            # 解决方案：完全忽略环境返回的reward，使用我们基于距离和惩罚机制计算的reward
            # 这样网络就会根据我们设计的reward信号来学习，而不是根据goal标志
            
            print(f'catch_flag: {catch_flag}')
            print(f'done: {done}')
            print(f'【调试】环境返回: reward_env={reward_env:.2f}, goal={goal}, good={good}')
            
            #if count == 1:  # 如果计数器为1 
            #    gps1, gps2, gps3, gps4, foot_gps1 = env.print_gps()  # 获取GPS位置
            #    x1 = gps_goal[0] - gps1[1]  # 计算目标位置与当前位置的差值
            #   y1 = gps_goal[1] - gps1[2]
            #  if x1 > -0.03 and y1 < 0.03:
            #        reward1 = 1  # 奖励为1
            #    elif -0.05 < x1 < -0.03 and 0.03 < y1 < 0.05:
            #        reward1 = 1  # 奖励为1
            #    else:
            #        reward1 = 0  # 奖励为0
            #    reward = reward1  # 奖励为reward1
            # === 【修复】改进的距离奖励：区分正确目标和错误目标 ===
            gps1, _, _, _, _ = env.print_gps()
            # 安全检查：确保gps1有足够的元素
            if len(gps1) < 3:
                print(f"警告：gps1长度不足 ({len(gps1)} < 3)，使用默认值")
                dx = 0.0
                dy = 0.0
            else:
                dx = gps_goal[0] - gps1[1]
                dy = gps_goal[1] - gps1[2]
            current_distance = (dx**2 + dy**2)**0.5

            # 距离变化奖励（鼓励靠近目标）
            success_flag1 = env.darwin.get_touch_sensor_value('grasp_L1_2')
            if prev_distance is not None:
                reward = (prev_distance - current_distance) * 5.0  # 放大系数可调
            else:
                reward = -current_distance  # 初始奖励

            prev_distance = current_distance  # 更新

            # 稀疏奖励：到达目标附近额外加分
            # 【修复】使用更全面的抓取判断：检查所有抓取传感器
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
            # 抓取成功：左右两侧都有传感器触发
            success_flag1 = 1 if (left_any and right_any) else 0
            
            if success_flag1 == 1:                       # 抓到了
    # 用你前面算好的 current_distance 即可
                if current_distance <= 0.04:             # 4 cm 容忍
                    reward += 300                              
                    print("✅ 抓到目标梯级，发放大奖励！")
                else:                            
                    reward -= 160                         # 抓错梯子，无大奖励
                    print("⚠️  抓到非目标梯级，无大奖励")
            if done == 1 and steps <6 and success_flag1 != 1:
                print("错误抓取！给予较大惩罚！")
                reward -= 100
            if done == 1 and steps >=6 and success_flag1 != 1:
                print("错误抓取！给予较大惩罚！")
                reward -= 100
            if done == 1 and steps <= 2 and success_flag1 != 1:
                print("因环境不稳定导致无效数据，跳过此步骤！！！")
                break
            reward -= 10
            return_all = return_all + reward  # 总奖励为当前奖励加上之前的总奖励
            
            steps += 1  # 步数加1
            next_obs_img, next_obs_tensor = env.get_img(steps, imgs)  # 获取下一个图像和图像张量
            next_obs = [next_obs_img, next_state]
            # print('获取下一个状态更新完毕')
            # === 【关键修复】移除样本过滤，确保所有样本都参与学习 ===
            # 问题：之前的条件存储（good==1）可能导致网络偏向学习"达到goal"而不是"最大化reward"
            # 解决方案：存储所有样本，让网络根据reward值来学习，而不是根据goal标志
            # 注意：标准PPO应该存储所有样本，这样才能正确学习奖励信号
            
            # 【修复】只跳过明显无效的样本（环境初始化问题）
            should_store = True
            if done == 1 and steps <= 2 and success_flag1 != 1:
                # 环境不稳定导致的无效数据，跳过
                should_store = False
                print(f"  跳过无效样本：done={done}, steps={steps}, success={success_flag1}")
            
            if should_store:
                # === 【关键修复】确保存储的reward是我们计算的reward，而不是环境返回的reward_env ===
                # 记录所有步，避免采样偏置（PPO 需要 on-policy 全覆盖）
                # 【调试信息】显示reward和goal的关系，验证reward是否独立于goal
                if steps % 10 == 0 or abs(reward) > 50:  # 每10步或大奖励时打印
                    print(f"【存储样本】reward={reward:.2f}, goal={goal}, reward_env={reward_env:.2f}, 差值={reward - reward_env:.2f}")
                
                ppo_shoulder.store_transition_catch(
                    state=[obs_img, robot_state, robot_state],
                    action=action_shoulder,       # 只传自己的动作
                    reward=reward,  # 【关键】使用我们计算的reward，而不是环境返回的reward_env
                    next_state=[next_obs_img, next_state, next_state],
                    done=done,
                    value=value_shoulder,         # 只传自己的价值
                    log_prob=log_prob_shoulder,  # 只传自己的对数概率
                    action_raw=action_raw_shoulder  # 传递原始动作用于正确计算 log_prob
                )
                ppo_arm.store_transition_catch(
                    state=[obs_img, robot_state, robot_state],
                    action=action_arm,           # 只传自己的动作
                    reward=reward,
                    next_state=[next_obs_img, next_state, next_state],
                    done=done,
                    value=value_arm,             # 只传自己的价值
                    log_prob=log_prob_arm,       # 只传自己的对数概率
                    action_raw=action_raw_arm     # 传递原始动作用于正确计算 log_prob
                )
            else:
                # 不存储此样本（用于调试，可以打印信息）
                if steps % 10 == 0:  # 每10步打印一次，避免输出过多
                    print(f"  跳过存储：good={good}, reward={reward:.2f}, steps={steps}")
            # ppo_catch.store_transition_catch(
            #     state=[obs_img, robot_state, robot_state],
            #     action_shoulder=action_shoulder,
            #     action_arm=action_arm,
            #     reward=reward,
            #     next_state=[next_obs_img, next_state, next_state],
            #     done=done,
            #     value_shoulder=value_combined, # 使用合并后的价值
            #     value_arm=value_combined,
            #     log_prob_shoulder=log_prob_combined, # 使用合并后的log_prob
            #     log_prob_arm=log_prob_combined
            # )
            robot_state = env.get_robot_state()  # 获取机器人状态



            obs_tensor = next_obs_tensor  # 更新图像张量
            
            if done == 1 or steps >= max_steps_per_episode:
                # === 【关键修复】在episode结束时，确保reward完全独立于goal ===
                # 问题：如果环境返回的goal=1但实际抓错了，我们需要确保reward是大惩罚
                # 解决方案：在episode结束时，根据实际抓取结果重新计算reward，完全忽略goal标志
                
                # 1. 每10个episode才调用learn()进行模型更新
                learn_counter += 1
                if learn_counter >= LEARN_INTERVAL:
                    print(f"\n--- Episode {i} 结束，已累积 {learn_counter} 个episode，开始学习 ---")
                    # 【优化】检查是否有数据可学习（避免空数据学习）
                    if len(ppo_shoulder.rewards) == 0 or len(ppo_arm.rewards) == 0:
                        print(f"  警告：Episode {i} 没有数据可学习（可能所有样本都被跳过），跳过本次学习")
                        loss = 0.0
                    else:
                        loss_shoulder = ppo_shoulder.learn()
                        loss_arm = ppo_arm.learn()
                        loss = loss_shoulder + loss_arm   
                        print('loss_arm:', loss_arm)
                        print('loss_shoulder:', loss_shoulder)
                        print('loss:', loss)
                    log_writer_catch.add(loss=loss)
                    learn_counter = 0  # 重置计数器，再等10个episode
                else:
                    print(f"--- Episode {i} 结束，累积中... ({learn_counter}/{LEARN_INTERVAL} episodes) ---")
                    loss = 0.0  # 未学习时loss为0
                    log_writer_catch.add(loss=loss)

                # 2. 准备好通用的 checkpoint 数据，避免重复写
                base_checkpoint_data = {
                    # --- 【修正】只保存一个策略网络的状态 ---
                    'policy_shoulder': ppo_shoulder.policy.state_dict(),
                    'optimizer_shoulder': ppo_shoulder.optimizer.state_dict(),
                    'policy_arm': ppo_arm.policy.state_dict(),
                    'optimizer_arm': ppo_arm.optimizer.state_dict(),
                    'episode': i
                }

                 # --- 【逻辑决策点】决定何时进行模型评估 ---
                is_checkpoint_interval = (i % CHECKPOINT_INTERVAL == 0) and (i != 0)
                # is_checkpoint_interval = i % CHECKPOINT_INTERVAL == 0
                # 我们使用 "是否到达检查点" 作为触发模型评估的唯一条件
                if is_checkpoint_interval:
                    print(f"\n--- 周期 {i}: 到达检查点，开始在当前环境进行模型测试 (共 {NUM_TEST_EPISODES} 轮有效) ---")
                    ppo_shoulder.policy.eval()
                    ppo_arm.policy.eval()

                    successful_test_episodes = 0
                    valid_test_cnt   = 0          # 已经跑完的有效轮次
                    total_test_cnt   = 0          # 总共开的轮次（含无效）
                    max_steps_per_test_episode = 500
                     

                    while valid_test_cnt < NUM_TEST_EPISODES:          # 只认有效轮次
                        total_test_cnt += 1
                        print(f"————————————————测试轮次 {valid_test_cnt + 1}/{NUM_TEST_EPISODES} "
                            f"(总开启 {total_test_cnt})——————————————")

                        # -------- 1. 初始化 --------
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
                            continue                 # 直接重开一轮

                        # -------- 2. 跑 episode --------
                        test_steps, test_done = 0, False
                        test_imgs = []
                        while not test_done and test_steps < max_steps_per_test_episode:
                            test_obs_img, test_obs_tensor = env.get_img(test_steps, test_imgs)
                            test_robot_state = env.get_robot_state()
                            if len(test_robot_state) < 6:
                                print("  测试警告：robot_state 长度不足，提前结束本轮。")
                                break

                            test_obs = (test_obs_tensor, test_robot_state)
                            with torch.no_grad():
                                action_shoulder_t, _, _, _ = ppo_shoulder.choose_action(
                                    episode_num=i, obs=test_obs, x_graph=test_robot_state, explore=False)
                                action_arm_t, _, _, _ = ppo_arm.choose_action(
                                    episode_num=i, obs=test_obs, x_graph=test_robot_state, explore=False)

                            # 转 float + 限幅
                            action_shoulder_t = float(action_shoulder_t[0]) if hasattr(action_shoulder_t, '__getitem__') else float(action_shoulder_t)
                            action_arm_t      = float(action_arm_t[0])      if hasattr(action_arm_t, '__getitem__')      else float(action_arm_t)
                            action_shoulder_t = np.clip(action_shoulder_t, -0.5, 0.5)
                            action_arm_t      = np.clip(action_arm_t, -0.5, 0.5)

                            test_gps1, test_gps2, test_gps3, test_gps4, _ = env.print_gps()
                            if len(test_gps1) < 3:
                                test_steps += 1
                                continue

                            test_catch_flag = 1.0 if test_steps >= 19 else 0.0
                            _, _, test_done_from_env, _, test_goal_from_env, _ = env.step(
                                test_robot_state, action_shoulder_t, action_arm_t, test_steps,
                                test_catch_flag, test_gps1, test_gps2, test_gps3, test_gps4,
                                f"test_img_{test_steps}.png")
                            if test_done_from_env or test_goal_from_env:
                                test_done = True
                            test_steps += 1

                        # -------- 3. 判定结果 --------
                        final_touch = env.darwin.get_touch_sensor_value('grasp_L1_2')
                        early_fail  = (test_steps <= 2 and final_touch != 1)
                        gps1, _, _, _, _ = env.print_gps()
                        # 安全检查：确保gps1有足够的元素
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

                        valid_test_cnt += 1          # 只有跑到这里才算完成一次有效测试
                    ppo_shoulder.policy.train()
                    ppo_arm.policy.train() # 两者都切换到训练模式
                    test_success_rate = successful_test_episodes
                    log_writer_catch.add(success_rate=test_success_rate)
                    print(f"\n--- 测试完成：{NUM_TEST_EPISODES}轮测试成功率为 {test_success_rate:.2f}% ---")
                    
                    # --- 【修正】排行榜也使用单一检查点 ---
                    model_ranking.add_and_manage(
                        new_score=test_success_rate,
                        new_checkpoint=base_checkpoint_data,
                        episode_id=i,
                        base_dir=path_list['model_path_catch_PPO']
                    )

                    model_ranking.print_current_rankings()

                else:
                    # 如果不是检查点周期，跳过测试
                    print(f"\n--- 周期 {i}: 未到达检查点，跳过模型测试 ---")

                # 3. 记录本轮训练日志并重置状态
                print(f"本轮训练累积奖励: {return_all:.2f}, 目标达成: {success_flag1}")
                log_writer_catch.add_log_prob_catch(log_prob_shoulder, log_prob_arm) 
                log_writer_catch.add_value_catch(value_shoulder, value_arm)
                # 使用 ppo_shoulder 或 ppo_arm 都可以，因为它们的 sigma 参数应该是一样的
                current_sigma = ppo_shoulder.get_current_sigma()
                log_writer_catch.add(sigma=current_sigma)
                log_writer_catch.add(return_all=return_all)
                log_writer_catch.add(goal=1 if success_flag1 else 0)
                log_writer_catch.clear()
                log_writer_catch.save_catch(log_file_latest_catch)
                
                # 4. 跳出while循环，开始下一个episode
                break

                
            #success_flag1 = env.darwin.get_touch_sensor_value('grasp_L1_2')

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
            log_writer_catch.clear()
            log_writer_catch.save_catch(log_file_latest_catch)  # 保存日志
            

        if success_flag1 == 1:
            success_catch += 1
            log_writer_catch.add(success_catch=success_catch)
            print("success_catch:", success_catch)
            print("抓取成功，开始抬腿训练...")
            total_episode = i
            print("tai_episoid:", tai_episoid)
            PPO_tai_episoid(ppo2_LegUpper=ppo2_LegUpper, ppo2_LegLower=ppo2_LegLower, ppo2_Ankle=ppo2_Ankle, existing_env=env, total_episode=total_episode, episode=tai_episoid, log_writer_tai=log_writer_tai, log_file_latest_tai=log_file_latest_tai)
            tai_episoid += 1 

    
    log_writer_catch.save_catch(log_file_latest_catch)  # 保存日志
    # 如果整个训练过程结束，返回抓取成功状态和环境实例
    return False, env