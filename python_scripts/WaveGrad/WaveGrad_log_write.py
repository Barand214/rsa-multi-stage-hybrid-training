import json
import os
import re
from datetime import datetime

import numpy as np


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if "torch.Tensor" in str(type(obj)):
            try:
                return obj.cpu().detach().numpy().tolist()
            except AttributeError:
                pass
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.void):
            return None
        return super().default(obj)


class Log_write:
    def __init__(self):
        self.data = {
            "start time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "episode_num": [],
            "shoulder_actions": [[]],
            "arm_actions": [[]],
            "LegUpper_actions": [[]],
            "LegLower_actions": [[]],
            "Ankle_actions": [[]],
            "loss": [],
            "diffusion_loss": [],
            "value_loss": [],
            "success_replay_size": [],
            "elite_replay_size": [],
            "policy_lr": [],
            "return_all": [],
            "goal": [],
            "safety_penalty": [],
            "rolling_success_rate_100": [],
            "rolling_mean_return_100": [],
            "test_episode": [],
            "test_grasp_success_rate": [],
            "test_score": [],
        }

    def add_action_catch(self, shoulder_action, arm_action):
        if not self.data["shoulder_actions"]:
            self.data["shoulder_actions"] = [[]]
        if not self.data["arm_actions"]:
            self.data["arm_actions"] = [[]]
        self.data["shoulder_actions"][-1].append(self._to_float(shoulder_action))
        self.data["arm_actions"][-1].append(self._to_float(arm_action))

    def add_action_tai(self, action_leg_upper, action_leg_lower, action_ankle):
        for key in ("LegUpper_actions", "LegLower_actions", "Ankle_actions"):
            if key not in self.data or not self.data[key]:
                self.data[key] = [[]]
        self.data["LegUpper_actions"][-1].append(self._to_float(action_leg_upper))
        self.data["LegLower_actions"][-1].append(self._to_float(action_leg_lower))
        self.data["Ankle_actions"][-1].append(self._to_float(action_ankle))

    def add(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def clear(self):
        for key in ("shoulder_actions", "arm_actions", "LegUpper_actions", "LegLower_actions", "Ankle_actions"):
            if key not in self.data or not self.data[key]:
                self.data[key] = [[]]
            elif self.data[key][-1]:
                self.data[key].append([])

    def save_catch(self, file_path):
        self._save(file_path, stage="catch")

    def save_tai(self, file_path):
        self._save(file_path, stage="tai")

    def _save(self, file_path, stage):
        data_to_save = json.loads(json.dumps(self.data, cls=CustomJSONEncoder))
        self._trim_empty_action_tail(data_to_save)
        data_to_save = self._filter_metrics(data_to_save, stage=stage)
        json_data = json.dumps(data_to_save, cls=CustomJSONEncoder, indent=4, ensure_ascii=False)
        pattern = r"\[\s*(-?\d+\.?\d*(?:\s*,\s*-?\d+\.?\d*)*)\s*\]"

        def compress_list(match):
            text = re.sub(r"\s+", " ", match.group(0))
            text = re.sub(r"\s*,\s*", ", ", text)
            text = re.sub(r"\[\s+", "[", text)
            text = re.sub(r"\s+\]", "]", text)
            return text.strip()

        formatted_json = re.sub(pattern, compress_list, json_data, flags=re.MULTILINE)
        tmp_file_path = f"{file_path}.tmp"
        with open(tmp_file_path, "w", encoding="utf-8") as file:
            file.write(formatted_json)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_file_path, file_path)
        print(f"WaveGrad log saved: {file_path}")

    def _filter_metrics(self, data_to_save, stage):
        common_keys = [
            "start time",
            "episode_num",
            "return_all",
            "goal",
            "loss",
            "diffusion_loss",
            "value_loss",
            "success_replay_size",
            "elite_replay_size",
            "policy_lr",
            "rolling_success_rate_100",
            "rolling_mean_return_100",
        ]
        catch_keys = [
            "shoulder_actions",
            "arm_actions",
            "safety_penalty",
            "test_episode",
            "test_grasp_success_rate",
            "test_score",
        ]
        tai_keys = [
            "LegUpper_actions",
            "LegLower_actions",
            "Ankle_actions",
        ]
        keys = common_keys + (catch_keys if stage == "catch" else tai_keys)
        return {key: data_to_save[key] for key in keys if key in data_to_save}

    def _trim_empty_action_tail(self, data_to_save):
        for key in ("shoulder_actions", "arm_actions", "LegUpper_actions", "LegLower_actions", "Ankle_actions"):
            values = data_to_save.get(key)
            if isinstance(values, list) and values and values[-1] == []:
                values.pop()

    def _to_float(self, value):
        if hasattr(value, "item") and callable(value.item):
            try:
                value = value.item()
            except Exception:
                pass
        return float(value)
