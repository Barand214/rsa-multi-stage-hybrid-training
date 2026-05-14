import numpy as np
import json
import re
from datetime import datetime
# import torch # 如果你想使用 isinstance 检查张量，请取消注释

class CustomJSONEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，处理 NumPy 数组、PyTorch 张量和 datetime 对象"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # elif isinstance(obj, torch.Tensor): # 如果环境中确定有 torch，这是更健壮的方式
        #     return obj.cpu().detach().numpy().tolist()
        elif "torch.Tensor" in str(type(obj)): # 使用字符串检查作为备选方案
            # 确保在调用 .cpu() 和 .detach() 之前对象确实是张量
            try:
                return obj.cpu().detach().numpy().tolist()
            except AttributeError:
                # 如果对象声称是 Tensor 但没有这些方法，按原样处理
                pass
        elif isinstance(obj, datetime):
            return obj.isoformat()
        # 处理 Python 的基本数字类型，以防万一传入的是 NumPy 数字类型
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
            return None # 或者选择其他合适的表示方式
        return super().default(obj)

class Log_write:
    def __init__(self):
        """初始化日志记录器"""
        self.data = {
            'start time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'episode_num': [],
            #'action_list': [[]],  # 初始化动作列表，包含一个空列表用于第一个序列
            'shoulder_actions': [[]],  # 存储肩膀动作
            'arm_actions': [[]],        # 存储手臂动作
            'log_prob_list': [[]],  # 初始化对数概率列表
            'value_list': [[]],  # 初始化状态价值列表
            # 分别记录三个智能体的loss值
            'loss_shoulder': [],  # 肩部智能体loss
            'loss_arm': [],       # 臂部智能体loss
            'loss_hppo': [],      # 离散门控智能体loss
            'loss_hppo_01': [],
            'loss_total': [],     # 总loss
            # 抬腿阶段loss
            'loss_LegUpper': [],  # 上腿智能体loss
            'loss_LegLower': [],  # 下腿智能体loss
            'loss_Ankle': [],     # 踝关节智能体loss
            'loss_hppo_tai': [],  # 抬腿阶段离散门控智能体loss
            'loss_total_tai': [],  # 抬腿阶段总loss
            #HPPO需要记录的值
            'loss_discrete':[],
            'loss_continuous':[],

            'loss_discrete_tai': [],
            'loss_continuous_tai': []
        }
        # 注意：不再有 self.action_list 实例变量

    def add_action_catch(self, a_shoulder, a_arm):
        """
        将两个动作添加到当前活动的操作序列中。
        动作会被添加到 self.data['shoulder_actions']和self.data['arm_actions'] 的最后一个子列表中。
        """
        # 确保 shoulder_actions 存在且至少有一个子列表 (通常由 __init__ 保证)
        if 'shoulder_actions' not in self.data or not self.data['shoulder_actions']:
            self.data['shoulder_actions'] = [[]] # 安全措施，理论上不应触发

        # 确保 arm_actions 存在且至少有一个子列表 (通常由 __init__ 保证)
        if 'arm_actions' not in self.data or not self.data['arm_actions']:
            self.data['arm_actions'] = [[]] # 安全措施，理论上不应触发

        # 如果 shoulder_action 是 Tensor 或 NumPy scalar，获取其 Python 基本类型值
        shoulder_action_item = a_shoulder
        if hasattr(a_shoulder, 'item') and callable(a_shoulder.item):
            try:
                shoulder_action_item = a_shoulder.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on action shoulder {a_shoulder}: {e}...73")
                # 保持 action 原样，或根据需要处理错误

        # 如果 arm_action 是 Tensor 或 NumPy scalar，获取其 Python 基本类型值
        arm_action_item = a_arm
        if hasattr(a_arm, 'item') and callable(a_arm.item):
            try:
                arm_action_item = a_arm.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on action arm {a_arm}: {e}...82")
                # 保持 action 原样，或根据需要处理错误

        # 检查转换后的类型是否为基本数字类型（int 或 float）
        if not isinstance(shoulder_action_item, (int, float)):
            print(f"Warning: Adding non-standard action type '{type(shoulder_action_item).__name__}' to action list. Value: {shoulder_action_item}")

        # 检查转换后的类型是否为基本数字类型（int 或 float）
        if not isinstance(arm_action_item, (int, float)):
            print(f"Warning: Adding non-standard action type '{type(arm_action_item).__name__}' to action list. Value: {arm_action_item}")

        # 将动作添加到最后一个（当前活动的）序列中
        self.data['shoulder_actions'][-1].append(shoulder_action_item)
        self.data['arm_actions'][-1].append(arm_action_item)
        
    def add_action_tai(self, action_LegUpper, action_LegLower, action_Ankle):
        """
        将三个动作添加到当前活动的操作序列中。
        动作会被添加到 self.data['LegUpper_actions']和self.data['LegLower_actions']和self.data['Ankle_actions'] 的最后一个子列表中。
        """
        # 确保所有动作列表都存在且至少有一个子列表
        for action_type in ['LegUpper_actions', 'LegLower_actions', 'Ankle_actions']:
            if action_type not in self.data or not self.data[action_type]:
                self.data[action_type] = [[]]  # 安全措施，理论上不应触发

        # 如果 shoulder_action 是 Tensor 或 NumPy scalar，获取其 Python 基本类型值
        LegUpper_action_item = action_LegUpper
        if hasattr(action_LegUpper, 'item') and callable(action_LegUpper.item):
            try:
                LegUpper_action_item = action_LegUpper.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on action LegUpper {action_LegUpper}: {e}...113")
                # 保持 action 原样，或根据需要处理错误

        # 如果 arm_action 是 Tensor 或 NumPy scalar，获取其 Python 基本类型值
        LegLower_action_item = action_LegLower
        if hasattr(action_LegLower, 'item') and callable(action_LegLower.item):
            try:
                LegLower_action_item = action_LegLower.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on action LegLower {action_LegLower}: {e}...122")
                # 保持 action 原样，或根据需要处理错误

        # 如果 Ankle_action 是 Tensor 或 NumPy scalar，获取其 Python 基本类型值
        Ankle_action_item = action_Ankle
        if hasattr(action_Ankle, 'item') and callable(action_Ankle.item):
            try:
                Ankle_action_item = action_Ankle.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on action Ankle {action_Ankle}: {e}...129")
                # 保持 action 原样，或根据需要处理错误

        # 检查转换后的类型是否为基本数字类型（int 或 float）
        for item, name in [(LegUpper_action_item, 'LegUpper_action_item'),
                          (LegLower_action_item, 'LegLower_action_item'),
                          (Ankle_action_item, 'Ankle_action_item')]:
            if not isinstance(item, (int, float)):
                print(f"Warning: Adding non-standard action type '{type(item).__name__}' to action list. Value: {item}")
       
        # 将动作添加到最后一个（当前活动的）序列中
        self.data['LegUpper_actions'][-1].append(LegUpper_action_item)
        self.data['LegLower_actions'][-1].append(LegLower_action_item)
        self.data['Ankle_actions'][-1].append(Ankle_action_item)

    def add_log_prob_catch(self, log_prob_shoulder, log_prob_arm):
        """
        将对数概率添加到当前活动的序列中。
        对数概率会被添加到 self.data['log_prob_list_shoulder'] 的最后一个子列表中。
        """
        # 确保 log_prob_list 存在且至少有一个子列表
        if 'log_prob_list_shoulder' not in self.data or not self.data['log_prob_list_shoulder']:
            self.data['log_prob_list_shoulder'] = [[]]
        if 'log_prob_list_arm' not in self.data or not self.data['log_prob_list_arm']:
            self.data['log_prob_list_arm'] = [[]]
            
        # 如果 log_prob 是 Tensor 或 NumPy scalar，获取其 Python 基本类型值
        log_prob_item = log_prob_shoulder
        if hasattr(log_prob_shoulder, 'item') and callable(log_prob_shoulder.item):
            try:
                log_prob_item = log_prob_shoulder.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on log_prob shoulder {log_prob_shoulder}: {e}...163")
                
        # 将对数概率添加到最后一个序列中
        self.data['log_prob_list_shoulder'][-1].append(float(log_prob_item))
        
        # 如果 log_prob 是 Tensor 或 NumPy scalar，获取其 Python 基本类型值
        log_prob_item = log_prob_arm
        if hasattr(log_prob_arm, 'item') and callable(log_prob_arm.item):
            try:
                log_prob_item = log_prob_arm.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on log_prob arm {log_prob_arm}: {e}...176")
                
        # 将对数概率添加到最后一个序列中
        self.data['log_prob_list_arm'][-1].append(float(log_prob_item))

    def add_chunk_log_prob_catch(self, log_prob):
        """
        记录 route3 抓取阶段每个 chunk 的对数概率。
        """
        if 'chunk_log_prob_catch' not in self.data:
            self.data['chunk_log_prob_catch'] = []

        log_prob_item = log_prob
        if hasattr(log_prob, 'item') and callable(log_prob.item):
            try:
                log_prob_item = log_prob.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on chunk log_prob catch {log_prob}: {e}")

        self.data['chunk_log_prob_catch'].append(float(log_prob_item))

    def add_log_prob_tai(self, log_prob_LegUpper, log_prob_LegLower, log_prob_Ankle):
        """
        将对数概率添加到当前活动的序列中。
        对数概率会被添加到 self.data['log_prob_list_LegUpper'] 的最后一个子列表中。
        """
        # 确保 log_prob_list 存在且至少有一个子列表
        if 'log_prob_list_LegUpper' not in self.data or not self.data['log_prob_list_LegUpper']:
            self.data['log_prob_list_LegUpper'] = [[]]
        if 'log_prob_list_LegLower' not in self.data or not self.data['log_prob_list_LegLower']:
            self.data['log_prob_list_LegLower'] = [[]]
        if 'log_prob_list_Ankle' not in self.data or not self.data['log_prob_list_Ankle']:
            self.data['log_prob_list_Ankle'] = [[]]
            
        # 如果 log_prob 是 Tensor 或 NumPy scalar，获取其 Python 基本类型值
        log_prob_item = log_prob_LegUpper
        if hasattr(log_prob_LegUpper, 'item') and callable(log_prob_LegUpper.item):
            try:
                log_prob_item = log_prob_LegUpper.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on log_prob LegUpper {log_prob_LegUpper}: {e}...196")
                
        # 将对数概率添加到最后一个序列中
        self.data['log_prob_list_LegUpper'][-1].append(float(log_prob_item))
        
        # 如果 log_prob 是 Tensor 或 NumPy scalar，获取其 Python 基本类型值
        log_prob_item = log_prob_LegLower
        if hasattr(log_prob_LegLower, 'item') and callable(log_prob_LegLower.item):
            try:
                log_prob_item = log_prob_LegLower.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on log_prob LegLower {log_prob_LegLower}: {e}...206")
                
        # 将对数概率添加到最后一个序列中
        self.data['log_prob_list_LegLower'][-1].append(float(log_prob_item))

        # 如果 log_prob 是 Tensor 或 NumPy scalar，获取其 Python 基本类型值
        log_prob_item = log_prob_Ankle
        if hasattr(log_prob_Ankle, 'item') and callable(log_prob_Ankle.item):
            try:
                log_prob_item = log_prob_Ankle.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on log_prob Ankle {log_prob_Ankle}: {e}...220")
                
        # 将对数概率添加到最后一个序列中
        self.data['log_prob_list_Ankle'][-1].append(float(log_prob_item))

    def add_chunk_log_prob_tai(self, log_prob):
        """
        记录 route3 抬腿阶段每个 chunk 的对数概率。
        """
        if 'chunk_log_prob_tai' not in self.data:
            self.data['chunk_log_prob_tai'] = []

        log_prob_item = log_prob
        if hasattr(log_prob, 'item') and callable(log_prob.item):
            try:
                log_prob_item = log_prob.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on chunk log_prob tai {log_prob}: {e}")

        self.data['chunk_log_prob_tai'].append(float(log_prob_item))

    def add_value_catch(self, value_shoulder, value_arm):
        """
        将状态价值添加到当前活动的序列中。
        状态价值会被添加到 self.data['value_list'] 的最后一个子列表中。
        """
        # 确保 value_list 存在且至少有一个子列表
        if 'value_list_shoulder' not in self.data or not self.data['value_list_shoulder']:
            self.data['value_list_shoulder'] = [[]]
        if 'value_list_arm' not in self.data or not self.data['value_list_arm']:
            self.data['value_list_arm'] = [[]]
            
        # 如果 value 是 Tensor 或 NumPy scalar，获取其 Python 基本类型值
        value_item = value_shoulder
        if hasattr(value_shoulder, 'item') and callable(value_shoulder.item):
            try:
                value_item = value_shoulder.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on value shoulder {value_shoulder}: {e}...242")
                
        # 将状态价值添加到最后一个序列中
        self.data['value_list_shoulder'][-1].append(float(value_item))

        # 如果 value 是 Tensor 或 NumPy scalar，获取其 Python 基本类型值
        value_item = value_arm
        if hasattr(value_arm, 'item') and callable(value_arm.item):
            try:
                value_item = value_arm.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on value arm {value_arm}: {e}...256")
                
        # 将状态价值添加到最后一个序列中
        self.data['value_list_arm'][-1].append(float(value_item))

    def add_chunk_value_catch(self, value):
        """
        记录 route3 抓取阶段每个 chunk 的价值估计。
        """
        if 'chunk_value_catch' not in self.data:
            self.data['chunk_value_catch'] = []

        value_item = value
        if hasattr(value, 'item') and callable(value.item):
            try:
                value_item = value.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on chunk value catch {value}: {e}")

        self.data['chunk_value_catch'].append(float(value_item))

    def add_value_tai(self, value_LegUpper, value_LegLower, value_Ankle):
        """
        将状态价值添加到当前活动的序列中。
        状态价值会被添加到 self.data['value_list'] 的最后一个子列表中。
        """
        # 确保 value_list 存在且至少有一个子列表
        if 'value_list_LegUpper' not in self.data or not self.data['value_list_LegUpper']:
            self.data['value_list_LegUpper'] = [[]]
        if 'value_list_LegLower' not in self.data or not self.data['value_list_LegLower']:
            self.data['value_list_LegLower'] = [[]]
        if 'value_list_Ankle' not in self.data or not self.data['value_list_Ankle']:
            self.data['value_list_Ankle'] = [[]]
                
        # 如果 value 是 Tensor 或 NumPy scalar，获取其 Python 基本类型值
        value_item = value_LegUpper
        if hasattr(value_LegUpper, 'item') and callable(value_LegUpper.item):
            try:
                value_item = value_LegUpper.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on value LegUpper {value_LegUpper}: {e}...277")
                
        # 将状态价值添加到最后一个序列中
        self.data['value_list_LegUpper'][-1].append(float(value_item))

        # 如果 value 是 Tensor 或 NumPy scalar，获取其 Python 基本类型值
        value_item = value_LegLower
        if hasattr(value_LegLower, 'item') and callable(value_LegLower.item):
            try:
                value_item = value_LegLower.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on value LegLower {value_LegLower}: {e}...292")
                
        # 将状态价值添加到最后一个序列中
        self.data['value_list_LegLower'][-1].append(float(value_item))

        # 如果 value 是 Tensor 或 NumPy scalar，获取其 Python 基本类型值
        value_item = value_Ankle
        if hasattr(value_Ankle, 'item') and callable(value_Ankle.item):
            try:
                value_item = value_Ankle.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on value Ankle {value_Ankle}: {e}...306")
                
        # 将状态价值添加到最后一个序列中
        self.data['value_list_Ankle'][-1].append(float(value_item))

    def add_chunk_value_tai(self, value):
        """
        记录 route3 抬腿阶段每个 chunk 的价值估计。
        """
        if 'chunk_value_tai' not in self.data:
            self.data['chunk_value_tai'] = []

        value_item = value
        if hasattr(value, 'item') and callable(value.item):
            try:
                value_item = value.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on chunk value tai {value}: {e}")

        self.data['chunk_value_tai'].append(float(value_item))

    def add_loss_hppo_catch(self,loss1,loss2):
        if hasattr(loss1, 'item') and hasattr(loss2,'item') and callable(loss1.item) and callable(loss2.item):
            try:
                loss1 = loss1.item()
                loss2 = loss2.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on loss_hppo {loss1}or{loss2}: {e}")
        self.data['loss_discrete'].append(float(loss1))
        self.data['loss_continuous'].append(float(loss2))

    def add_loss_hppo_tai(self,loss1,loss2):
        if hasattr(loss1, 'item') and hasattr(loss2,'item') and callable(loss1.item) and callable(loss2.item):
            try:
                loss1 = loss1.item()
                loss2 = loss2.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on loss_hppo {loss1}or{loss2}: {e}")
        self.data['loss_discrete'].append(float(loss1))
        self.data['loss_continuous'].append(float(loss2))

    def add_loss_catch(self, loss_shoulder, loss_arm, loss_hppo, loss_total):
        """
        记录抓取阶段三个智能体的loss值
        Args:
            loss_shoulder: 肩部智能体loss
            loss_arm: 臂部智能体loss  
            loss_hppo: 离散门控智能体loss
            loss_total: 总loss
        """
        # 处理Tensor类型
        if hasattr(loss_shoulder, 'item') and callable(loss_shoulder.item):
            try:
                loss_shoulder = loss_shoulder.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on loss_shoulder {loss_shoulder}: {e}")
        
        if hasattr(loss_arm, 'item') and callable(loss_arm.item):
            try:
                loss_arm = loss_arm.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on loss_arm {loss_arm}: {e}")
                
        if hasattr(loss_hppo, 'item') and callable(loss_hppo.item):
            try:
                loss_hppo = loss_hppo.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on loss_hppo {loss_hppo}: {e}")
                
        if hasattr(loss_total, 'item') and callable(loss_total.item):
            try:
                loss_total = loss_total.item()
            except Exception as e:
                print(f"Warning: Could not call .item() on loss_total {loss_total}: {e}")
        
        # 记录loss值
        self.data['loss_shoulder'].append(float(loss_shoulder))
        self.data['loss_arm'].append(float(loss_arm))
        self.data['loss_hppo'].append(float(loss_hppo))
        self.data['loss_total'].append(float(loss_total))

    def add_loss_tai(self, loss_LegUpper, loss_LegLower, loss_Ankle, loss_hppo, loss_total):
        """
        记录抬腿阶段三个智能体的loss值
        Args:
            loss_LegUpper: 上腿智能体loss
            loss_LegLower: 下腿智能体loss
            loss_Ankle: 踝关节智能体loss
            loss_hppo: 离散门控智能体loss
            loss_total: 总loss
        """
        # 处理Tensor类型
        for loss_name, loss_value in [('LegUpper', loss_LegUpper), ('LegLower', loss_LegLower), 
                                     ('Ankle', loss_Ankle), ('hppo', loss_hppo), ('total', loss_total)]:
            if hasattr(loss_value, 'item') and callable(loss_value.item):
                try:
                    loss_value = loss_value.item()
                except Exception as e:
                    print(f"Warning: Could not call .item() on loss_{loss_name} {loss_value}: {e}")
        
        # 记录loss值
        self.data['loss_LegUpper'].append(float(loss_LegUpper))
        self.data['loss_LegLower'].append(float(loss_LegLower))
        self.data['loss_Ankle'].append(float(loss_Ankle))
        self.data['loss_hppo_tai'].append(float(loss_hppo))
        self.data['loss_total_tai'].append(float(loss_total))

    def add(self, **kwargs):
        """
        向日志添加任意键值对数据。
        如果键不存在，则创建它并初始化为空列表，然后追加值。
        """
        for key, value in kwargs.items():
            # 确保我们不意外地覆盖核心列表，如 'action_list'
            if key == 'action_list':
                 print(f"Warning: Attempted to use reserved key 'action_list' in add(). Please use add_action() for actions.")
                 continue
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def clear(self):
        """
        标记当前序列的结束（例如，一个 episode 结束）。
        准备开始记录下一个新的序列。
        会在所有列表中添加一个新的空列表。
        """
        # 处理shoulder_actions
        if 'shoulder_actions' in self.data and self.data['shoulder_actions']:
            if self.data['shoulder_actions'][-1]: # 检查最后一个列表是否非空
                self.data['shoulder_actions'].append([])
        else:
            self.data['shoulder_actions'] = [[]]
            
        # 处理arm_actions
        if 'arm_actions' in self.data and self.data['arm_actions']:
            if self.data['arm_actions'][-1]: # 检查最后一个列表是否非空
                self.data['arm_actions'].append([])
        else:
            self.data['arm_actions'] = [[]]
            
        # 处理log_prob_list
        if 'log_prob_list_shoulder' in self.data and self.data['log_prob_list_shoulder']:
            if self.data['log_prob_list_shoulder'][-1]: # 检查最后一个列表是否非空
                self.data['log_prob_list_shoulder'].append([])
        else:
            self.data['log_prob_list_shoulder'] = [[]]
        # 处理log_prob_list
        if 'log_prob_list_arm' in self.data and self.data['log_prob_list_arm']:
            if self.data['log_prob_list_arm'][-1]: # 检查最后一个列表是否非空
                self.data['log_prob_list_arm'].append([])
        else:
            self.data['log_prob_list_arm'] = [[]]

        # 处理value_list
        if 'value_list_shoulder' in self.data and self.data['value_list_shoulder']:
            if self.data['value_list_shoulder'][-1]: # 检查最后一个列表是否非空
                self.data['value_list_shoulder'].append([])
        else:
            self.data['value_list_shoulder'] = [[]]
        # 处理value_list
        if 'value_list_arm' in self.data and self.data['value_list_arm']:
            if self.data['value_list_arm'][-1]: # 检查最后一个列表是否非空
                self.data['value_list_arm'].append([])
        else:
            self.data['value_list_arm'] = [[]]

    def get(self, key):
        """获取指定键的所有历史记录值（一个列表）"""
        return self.data.get(key, [])

    def _filter_eval_metrics(self, data_to_save, stage):
        """仅保留用于评估训练效果的关键指标。"""
        common_keys = [
            'start time',
            'episode_num',
            'return_all',
            'goal',
        ]

        catch_keys = [
            'grasp_steps',
            'grasp_decisions',
            'test_episode',
            'test_grasp_success_rate',
            'test_tai_success_rate',
            'test_pipeline_success_rate',
            'test_mean_total_return',
            'test_score',
        ]

        tai_keys = [
            'tai_steps',
            'tai_decisions',
        ]

        if stage == 'catch':
            ordered_keys = common_keys + catch_keys
        else:
            ordered_keys = common_keys + tai_keys

        filtered = {}
        for key in ordered_keys:
            if key in data_to_save:
                filtered[key] = data_to_save[key]
        return filtered

    def save_catch(self, file_path):
        """将当前所有日志数据保存到指定的 JSON 文件中"""
        print(f'保存日志...')

        # --- 数据准备 ---
        # 创建一个数据的深拷贝用于保存，以防修改影响原始数据
        # 使用 JSON 序列化和反序列化是一种简单的深拷贝方式，能处理嵌套结构
        try:
            data_to_save = json.loads(json.dumps(self.data, cls=CustomJSONEncoder))
        except Exception as e:
             print(f"Error creating deep copy of data for saving: {e}")
             # 可以选择是继续使用 self.data 还是中止保存
             data_to_save = self.data # 回退到使用原始数据（浅拷贝）

        data_to_save = self._filter_eval_metrics(data_to_save, stage='catch')

        # --- JSON 序列化 ---
        try:
            # 使用自定义编码器将准备好的数据转换为 JSON 字符串
            json_data = json.dumps(data_to_save, cls=CustomJSONEncoder, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error during JSON serialization: {e}")
            return # 无法序列化，中止保存

        #--- 格式化特定字段 ---
        # # 处理 episode_num 字段，将其格式化为多行
        # if 'episode_num' in data_to_save and len(data_to_save['episode_num']) > 10:
        #     # 将 episode_num 格式化为每行最多 10 个数字
        #     episode_chunks = []
        #     chunk_size = 20
        #     episodes = data_to_save['episode_num']
            
        #     for i in range(0, len(episodes), chunk_size):
        #         chunk = episodes[i:i+chunk_size]
        #         episode_chunks.append(json.dumps(chunk))
            
        #     # 创建格式化的 episode_num 字符串
        #     formatted_episodes = "[\n        " + ",\n        ".join(episode_chunks) + "\n    ]"
            
        #     # 在 JSON 字符串中替换 episode_num 部分
        #     pattern_episode = r'"episode_num":\s*\[\s*[\s\S]*?\]'
        #     json_data = re.sub(pattern_episode, f'"episode_num": {formatted_episodes}', json_data)
        
        # --- 正则表达式格式化 action_list ---
        # 改进的模式，尝试更灵活地匹配数字列表（整数）
        pattern = r'\[\s*(-?\d+\.?\d*(?:\s*,\s*-?\d+\.?\d*)*)\s*\]'

        def compress_list(match):
            # 获取匹配到的列表内容（包括括号）
            list_content = match.group(0)
            # 移除内部的所有换行符和多余的空白，替换为单个空格
            compressed_content = re.sub(r'\s+', ' ', list_content)
            # 标准化逗号后的空格
            compressed_content = re.sub(r'\s*,\s*', ', ', compressed_content)
            # 标准化括号内外的空格
            compressed_content = re.sub(r'\[\s+', '[', compressed_content)
            compressed_content = re.sub(r'\s+\]', ']', compressed_content)
            return compressed_content.strip()

        try:
            # 应用正则表达式替换，注意 re.MULTILINE 可能有助于跨行匹配
            formatted_json = re.sub(pattern, compress_list, json_data, flags=re.MULTILINE)
        except Exception as e:
            print(f"Error during regex formatting: {e}")
            formatted_json = json_data # 如果正则出错，回退到未格式化的 JSON

        # --- 文件写入 ---
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
            print(f"日志保存成功")
        except IOError as e:
            print(f"错误：无法写入日志文件: {e}")
        except Exception as e:
            print(f"保存过程中发生意外错误: {e}")

    def save_tai(self, file_path):
        """将当前所有日志数据保存到指定的 JSON 文件中"""
        print(f'保存日志...')

        # --- 数据准备 ---
        # 创建一个数据的深拷贝用于保存，以防修改影响原始数据
        # 使用 JSON 序列化和反序列化是一种简单的深拷贝方式，能处理嵌套结构
        try:
            data_to_save = json.loads(json.dumps(self.data, cls=CustomJSONEncoder))
        except Exception as e:
             print(f"Error creating deep copy of data for saving: {e}")
             # 可以选择是继续使用 self.data 还是中止保存
             data_to_save = self.data # 回退到使用原始数据（浅拷贝）

        data_to_save = self._filter_eval_metrics(data_to_save, stage='tai')

        # --- JSON 序列化 ---
        try:
            # 使用自定义编码器将准备好的数据转换为 JSON 字符串
            json_data = json.dumps(data_to_save, cls=CustomJSONEncoder, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error during JSON serialization: {e}")
            return # 无法序列化，中止保存

        #--- 格式化特定字段 ---
        # 处理 episode_num 字段，将其格式化为多行
        if 'episode_num' in data_to_save and len(data_to_save['episode_num']) > 10:
            # 将 episode_num 格式化为每行最多 10 个数字
            episode_chunks = []
            chunk_size = 20
            episodes = data_to_save['episode_num']
            
            for i in range(0, len(episodes), chunk_size):
                chunk = episodes[i:i+chunk_size]
                episode_chunks.append(json.dumps(chunk))
            
            # 创建格式化的 episode_num 字符串
            formatted_episodes = "[\n        " + ",\n        ".join(episode_chunks) + "\n    ]"
            
            # 在 JSON 字符串中替换 episode_num 部分
            pattern_episode = r'"episode_num":\s*\[\s*[\s\S]*?\]'
            json_data = re.sub(pattern_episode, f'"episode_num": {formatted_episodes}', json_data)
        
        # --- 正则表达式格式化 action_list ---
        # 改进的模式，尝试更灵活地匹配数字列表（整数）
        pattern = r'\[\s*(-?\d+\.?\d*(?:\s*,\s*-?\d+\.?\d*)*)\s*\]'

        def compress_list(match):
            # 获取匹配到的列表内容（包括括号）
            list_content = match.group(0)
            # 移除内部的所有换行符和多余的空白，替换为单个空格
            compressed_content = re.sub(r'\s+', ' ', list_content)
            # 标准化逗号后的空格
            compressed_content = re.sub(r'\s*,\s*', ', ', compressed_content)
            # 标准化括号内外的空格
            compressed_content = re.sub(r'\[\s+', '[', compressed_content)
            compressed_content = re.sub(r'\s+\]', ']', compressed_content)
            return compressed_content.strip()

        try:
            # 应用正则表达式替换，注意 re.MULTILINE 可能有助于跨行匹配
            formatted_json = re.sub(pattern, compress_list, json_data, flags=re.MULTILINE)
        except Exception as e:
            print(f"Error during regex formatting: {e}")
            formatted_json = json_data # 如果正则出错，回退到未格式化的 JSON

        # --- 文件写入 ---
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
            print(f"日志保存成功")
        except IOError as e:
            print(f"错误：无法写入日志文件: {e}")
        except Exception as e:
            print(f"保存过程中发生意外错误: {e}")