# Route 3：Webots 真正“双 Python 架构”骨架

## 架构目标

- **Webots controller 进程（Python 3.7.12）**：只负责仿真推进、传感器读取、动作执行、奖励/终止条件计算。
- **外部训练进程（Python 3.11.14）**：只负责 Diffusion Policy 推理、数据收集、在线 RL 更新、checkpoint 与实验记录。
- 两边通过 **TCP socket + length-prefix + pickle(protocol=4)** 通信。

> 之所以固定 `pickle protocol=4`，是因为它可以稳定跨 **Python 3.7** 和 **Python 3.11** 传输 numpy 数组。

---

## 文件说明

### 1) `WebotsEnvServer.py`
放到 **Webots controller** 目录下，作为 Webots 机器人 controller 运行。

暴露 RPC：
- `reset(seed=None, options=None)`
- `step_grasp(action)`
- `step_tai(action)`
- `close()`

特点：
- 自带最小 `DarwinAdapter`，**不再 import 旧 `Environment.step/step2`**。
- **不依赖旧 `RobotRun1/2`**。
- 保留了你现有项目的核心动作映射和两阶段传感器判断思路。

### 2) `dp_client.py`
外部训练端的通信客户端：
- 连接 Webots server
- 请求 `reset`
- 发送 `step_grasp / step_tai`
- 接收 `obs/reward/done/info`

### 3) `dp_model.py`
外部训练端的 Diffusion Policy 骨架：
- 视觉 + 状态 + graph encoder
- conditional diffusion actor
- critic
- rollout buffer
- PPO-style online update

### 4) `dp_online_trainer.py`
外部训练主循环：
- 两阶段训练调度
- 分段 rollout
- 在线更新 grasp/tai 两个 agent
- checkpoint
- metrics.jsonl / summary.json

---

## 依赖建议

### A. Webots controller（Python 3.7.12）
最少需要：
- Webots 自带 controller Python API
- `numpy`
- `Pillow`

建议：
```bash
python3.7 -m pip install numpy Pillow
```

### B. 外部训练进程（Python 3.11.14）
最少需要：
- `numpy`
- `torch`
- 可选：`torch_geometric`（不装也能跑，自动 fallback 到 MLP graph branch）

建议：
```bash
python3.11 -m pip install numpy torch
# 可选
python3.11 -m pip install torch-geometric
```

---

## 启动顺序

### 第一步：把 `WebotsEnvServer.py` 放进 controller 目录
例如：
```text
<your_webots_project>/controllers/WebotsEnvServer/WebotsEnvServer.py
```

然后在 `.wbt` 里把 Darwin 机器人的 controller 名设置成对应目录名。

### 第二步：启动 Webots 世界
打开 world 后，controller 会启动，并在控制台打印：
```text
[WebotsEnvServer] listening on 127.0.0.1:8765
```

### 第三步：在外部 Python 3.11.14 环境做 smoke test
```bash
python dp_client.py --host 127.0.0.1 --port 8765 --mode smoke
```

如果成功，说明双进程通信已经通了。

### 第四步：启动训练
```bash
python dp_online_trainer.py \
  --mode train \
  --host 127.0.0.1 \
  --port 8765 \
  --episodes 1000 \
  --save-dir runs/route3_dp_socket
```

### 第五步：加载 checkpoint 做评估
```bash
python dp_online_trainer.py \
  --mode eval \
  --host 127.0.0.1 \
  --port 8765 \
  --eval-episodes 50 \
  --grasp-ckpt runs/route3_dp_socket/checkpoints/grasp/grasp_episode_500.ckpt \
  --tai-ckpt runs/route3_dp_socket/checkpoints/tai/tai_episode_500.ckpt
```

---

## 与你现有工程对接时要注意的点

### 1. `gps_goal / gps_goal1`
`WebotsEnvServer.py` 会优先尝试从：
```python
python_scripts.Project_config
```
读取 `gps_goal` 和 `gps_goal1`。

如果你的项目里变量名不同，直接改这里的 import 或 fallback 常量即可。

### 2. 旧 reward 逻辑不是逐字复制，而是“按现有 DP trainer 实际训练语义复刻”
- 抓取阶段：按你现有 `DiffusionPolicy_two_stage_online_webots.py` 的 **外部 reward 逻辑** 走。
- 抬腿阶段：按你现有 `step2 + _compute_tai_reward(...)` 的整体语义走。

这比机械复制 `RobotRun1/2` 更接近你现在真正用来训练 Diffusion Policy 的行为。

### 3. Webots 端不再 import 旧 PPO / RobotRun1 / RobotRun2
这是 Route 3 的关键。

也就是说：
- **旧文件可以保留做 baseline**
- **新 server 完全走自己的 env step 实现**

---

## 如何做横向对比实验

建议至少做下面三组：

### A 组：旧单进程基线
- 旧 `DiffusionPolicy_two_stage_online_webots.py`
- 或旧 PPO / RobotRun1/2 流程

### B 组：新双 Python 架构（本骨架）
- `WebotsEnvServer.py` + `dp_online_trainer.py`

### C 组：新双 Python 架构 + 不同 chunk_len / update 频率
比如：
- grasp `chunk_len=4` vs `chunk_len=8`
- tai `chunk_len=3` vs `chunk_len=5`
- `update_every_episodes=5` vs `10`

---

## 横向对比时要控制住的变量

每组实验尽量保持一致：
- 同一个 Webots world
- 同一个 reset 随机范围
- 同一个 `grasp_trigger_step=19`
- 同样的最大阶段步数：
  - grasp = 120
  - tai = 21
- 同样的评估 episode 数
- 同样的随机种子集合，例如：
  - `42, 43, 44, 45, 46`

---

## 推荐记录指标

每个实验都记：
- `grasp_success_rate`
- `tai_success_rate`
- `pipeline_success_rate`
- `mean_total_return`
- 达到某个 success rate 所需 episode 数
- wall-clock 时间
- 每秒环境 step 数（吞吐）

### 推荐主指标
最关键的是：
- **pipeline_success_rate**
- **达到固定 pipeline_success_rate 的样本效率**

因为这两个最能体现 Route 3 是否真的值回票价。

---

## 目录建议

```text
route3_dual_python/
├── WebotsEnvServer.py
├── dp_client.py
├── dp_model.py
├── dp_online_trainer.py
└── README_route3.md
```

---

## 最后建议

先跑这三个阶段：

1. **通信 smoke test**：`reset -> step_grasp -> close`
2. **只跑 grasp 训练**：确认阶段 1 reward/成功逻辑稳定
3. **再开完整两阶段**：确认 `grasp_success -> step_tai` 衔接无误

如果你后面要继续，我最推荐的下一步是：
- 我可以再给你补一版 **“直接贴进你当前项目目录结构”的 integration 版**，把 import 路径和 Webots controller 目录一起对齐到你现在的工程布局。
