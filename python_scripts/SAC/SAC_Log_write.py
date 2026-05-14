import numpy as np
import json
import re
from datetime import datetime

class CustomJSONEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，处理 NumPy 数组、PyTorch 张量和 datetime 对象"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif "torch.Tensor" in str(type(obj)):
            try:
                return obj.cpu().detach().numpy().tolist()
            except AttributeError:
                pass
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return super().default(obj)

class SAC_Log_write:
    def __init__(self):
        """初始化SAC日志记录器"""
        self.data = {
            'start time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'save time': [],
            'episode_num': [],
            'action_list': [[]],
            'continuous_action_list': [[]],  # 记录连续动作
            'q_loss_list': [],              # 记录Q网络损失
            'policy_loss_list': [],         # 记录策略网络损失
            'alpha_loss_list': [],          # 记录温度参数损失
            'log_alpha_list': [],           # 记录温度参数对数值
            'alpha_list': [],               # 记录温度参数值
            'reward_list': [[]],            # 记录奖励序列
            'return_all_list': [],          # 记录累计奖励
            'steps_list': [],               # 记录步数
            'goal_list': []                 # 记录目标达成情况
        }

    def add_continuous_action(self, action):
        """记录连续动作值"""
        if 'continuous_action_list' not in self.data:
            self.data['continuous_action_list'] = [[]]
        self.data['continuous_action_list'][-1].append(action)

    def add_losses(self, q_loss, policy_loss, alpha_loss):
        """记录各种损失值"""
        self.data['q_loss_list'].append(q_loss)
        self.data['policy_loss_list'].append(policy_loss)
        self.data['alpha_loss_list'].append(alpha_loss)

    def add_alpha(self, log_alpha, alpha):
        """记录温度参数"""
        self.data['log_alpha_list'].append(log_alpha)
        self.data['alpha_list'].append(alpha)

    def add_reward(self, reward):
        """记录奖励"""
        if 'reward_list' not in self.data:
            self.data['reward_list'] = [[]]
        self.data['reward_list'][-1].append(reward)

    def add_return(self, return_all):
        """记录累计奖励"""
        self.data['return_all_list'].append(return_all)

    def add_step(self, step):
        """记录步数"""
        self.data['steps_list'].append(step)

    def add_goal(self, goal):
        """记录目标达成情况"""
        self.data['goal_list'].append(goal)

    def add_action(self, action):
        """记录离散动作"""
        if 'action_list' not in self.data or not self.data['action_list']:
            self.data['action_list'] = [[]]

        action_item = action
        if hasattr(action, 'item') and callable(action.item):
            try:
                action_item = action.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on action {action}: {e}")

        if not isinstance(action_item, (int, float)):
            print(f"Warning: Adding non-standard action type '{type(action_item).__name__}' to action list. Value: {action_item}")

        self.data['action_list'][-1].append(action_item)

    def add(self, **kwargs):
        """向日志添加任意键值对数据"""
        for key, value in kwargs.items():
            if key in ['action_list', 'continuous_action_list', 'reward_list']:
                print(f"Warning: Attempted to use reserved key '{key}' in add(). Please use the specific add method.")
                continue
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def clear(self):
        """标记当前序列的结束，准备开始新的序列"""
        for key in ['action_list', 'continuous_action_list', 'reward_list']:
            if key in self.data and self.data[key]:
                if self.data[key][-1]:  # 检查最后一个列表是否非空
                    self.data[key].append([])
            else:
                self.data[key] = [[]]

    def get(self, key):
        """获取指定键的所有历史记录值"""
        return self.data.get(key, [])

    def save_catch(self, file_path):
        """保存抓取阶段的日志"""
        self.data['save time'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(f'保存日志...')

        try:
            data_to_save = json.loads(json.dumps(self.data, cls=CustomJSONEncoder))
        except Exception as e:
            print(f"Error creating deep copy of data for saving: {e}")
            data_to_save = self.data

        # 修改episode_num，只保留最大值
        if 'episode_num' in data_to_save and data_to_save['episode_num']:
            max_episode = max(data_to_save['episode_num'])
            data_to_save['episode_num'] = [max_episode]

        # 移除末尾的空列表
        for key in ['action_list', 'continuous_action_list', 'reward_list']:
            if key in data_to_save and data_to_save[key] and not data_to_save[key][-1]:
                data_to_save[key] = data_to_save[key][:-1]

        try:
            json_data = json.dumps(data_to_save, cls=CustomJSONEncoder, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error during JSON serialization: {e}")
            return

        # 格式化数字列表
        pattern = r'\[\s*(-?\d+\.?\d*(?:\s*,\s*-?\d+\.?\d*)*)\s*\]'

        def compress_list(match):
            list_content = match.group(0)
            compressed_content = re.sub(r'\s+', ' ', list_content)
            compressed_content = re.sub(r'\s*,\s*', ', ', compressed_content)
            compressed_content = re.sub(r'\[\s+', '[', compressed_content)
            compressed_content = re.sub(r'\s+\]', ']', compressed_content)
            return compressed_content.strip()

        try:
            formatted_json = re.sub(pattern, compress_list, json_data, flags=re.MULTILINE)
        except Exception as e:
            print(f"Error during regex formatting: {e}")
            formatted_json = json_data

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
            print(f"日志保存成功")
        except IOError as e:
            print(f"错误：无法写入日志文件: {e}")
        except Exception as e:
            print(f"保存过程中发生意外错误: {e}")

    def save_tai(self, file_path):
        """保存抬腿阶段的日志"""
        self.data['save time'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(f'保存日志...')

        try:
            data_to_save = json.loads(json.dumps(self.data, cls=CustomJSONEncoder))
        except Exception as e:
            print(f"Error creating deep copy of data for saving: {e}")
            data_to_save = self.data

        # 移除末尾的空列表
        for key in ['action_list', 'continuous_action_list', 'reward_list']:
            if key in data_to_save and data_to_save[key] and not data_to_save[key][-1]:
                data_to_save[key] = data_to_save[key][:-1]

        try:
            json_data = json.dumps(data_to_save, cls=CustomJSONEncoder, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error during JSON serialization: {e}")
            return

        # 格式化数字列表
        pattern = r'\[\s*(-?\d+\.?\d*(?:\s*,\s*-?\d+\.?\d*)*)\s*\]'

        def compress_list(match):
            list_content = match.group(0)
            compressed_content = re.sub(r'\s+', ' ', list_content)
            compressed_content = re.sub(r'\s*,\s*', ', ', compressed_content)
            compressed_content = re.sub(r'\[\s+', '[', compressed_content)
            compressed_content = re.sub(r'\s+\]', ']', compressed_content)
            return compressed_content.strip()

        try:
            formatted_json = re.sub(pattern, compress_list, json_data, flags=re.MULTILINE)
        except Exception as e:
            print(f"Error during regex formatting: {e}")
            formatted_json = json_data

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
            print(f"日志保存成功")
        except IOError as e:
            print(f"错误：无法写入日志文件: {e}")
        except Exception as e:
            print(f"保存过程中发生意外错误: {e}") 