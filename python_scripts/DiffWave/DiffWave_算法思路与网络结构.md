# DiffWave 算法思路与网络结构说明

## 1. 总体思路

当前 DiffWave 训练链路的目标是用扩散策略替代原 PPO 策略，在 Webots 中完成机器人抓取和抬腿两阶段训练。整体设计不是传统 PPO 的 `log_prob + ratio + clip`，而是：

1. Webots 负责仿真、传感器读取、动作执行、奖励计算和 episode 流程。
2. Python 3.11 的 mission 环境作为 DiffWave GPU 服务，负责神经网络推理、经验缓存、学习和 checkpoint。
3. 每个 step 都根据当前图像、机器人状态、图结构状态和安全特征重新采样动作。
4. 训练时使用奖励加权 diffusion loss，同时加入 value head 和 action-value critic 作为辅助学习信号。
5. 成功轨迹和高回报轨迹会长期保留在 replay 中，避免稀有成功样本被单个 episode 结束后的 `clear_memory()` 遗忘。

核心入口关系如下：

```text
Webots Train_main.py
    -> DiffWave_episoid_1.py
        -> DiffWaveGPUClient
            -> mission python 启动 diffwave_gpu_service.py
                -> DiffWaveAgent / DiffWaveTaiAgent
                    -> DiffWavePolicy + ActionValueCritic
```

## 2. 双 Python + GPU 运行方式

Webots 控制器继续使用 Webots 自带/兼容的 Python 3.7 环境，因为 Webots 的 `controller` API 依赖该运行时。DiffWave 网络训练放在 `D:\anaconda\envs\mission\python.exe` 中运行，用于调用 Python 3.11 和 CUDA 版本 PyTorch。

Webots 侧通过 `python_scripts/DiffWave/diffwave_gpu_client.py` 自动启动 GPU 服务：

```text
D:\anaconda\envs\mission\python.exe -m python_scripts.DiffWave.diffwave_gpu_service
```

两边通过 socket + pickle protocol 4 通信，默认端口为 `127.0.0.1:8876`。Webots 是主控流程，GPU 服务只作为算法服务被调用。

主要 RPC 接口：

```text
choose_catch      根据当前观测返回 shoulder / arm 动作和值估计
store_catch       存储抓取阶段 transition
learn_catch       训练抓取阶段两个 agent
choose_tai        根据当前观测返回抬腿三个关节动作和值估计
store_tai         存储抬腿阶段 transition
learn_tai         训练抬腿阶段三个 agent
save_checkpoint   保存 policy、optimizer、critic、replay 状态
```

## 3. 训练数据流

抓取阶段每一步的数据流如下：

```text
1. Webots 读取当前 obs_img
2. Webots 读取 robot_state
3. Webots 构造 safety_features
4. GPU 服务调用 DiffWaveAgent.choose_action()
5. Webots 执行动作并计算 reward / done / success_flag
6. Webots 读取 next_obs_img 和 next_state
7. GPU 服务调用 store_transition_catch()
8. episode 结束后调用 learn_catch()
```

抬腿阶段同理，只是动作维度由两个独立 agent 变成三个独立 agent：

```text
leg_upper, leg_lower, ankle
```

当前实现没有缓存整段动作序列再执行，而是每个 step 都重新根据当前观测生成动作。这一点更接近 PPO/SAC 的在线决策方式，也能避免前几帧错误导致整段动作全部偏离。

## 4. 网络结构

主要网络定义在 `python_scripts/DiffWave/DiffWave_policy.py`。

### 4.1 FeatureEncoder

`FeatureEncoder` 负责把图像、机器人状态、图结构和安全特征融合成一个条件特征向量。

输入：

```text
image           当前摄像头图像，默认单通道
state           机器人状态向量，固定整理为 20 维
x_graph         图结构节点状态，默认 19 个节点
safety_features 安全特征，默认 14 维
```

图像分支：

```text
Conv2d(1 -> 32)
ReLU
Conv2d(32 -> 32)
ReLU
Conv2d(32 -> 32)
FC 6272 -> 6000 -> 100
min-max normalize
```

状态分支：

```text
state 20 维
FC 20 -> 100 -> 100
min-max normalize
```

图结构分支：

```text
GraphSAGE
GATConv
GraphSAGE
GATConv
GCNConv
mean pooling
FC 1000 -> 100
min-max normalize
```

融合方式：

```text
[image_feature, state_feature, graph_feature] -> 300 维
FC 300 -> 200
safety_features -> FC -> 200
最终条件特征 = 主融合特征 + safety 特征
```

最终输出：

```text
cond_features: [batch, 200]
```

### 4.2 DiffWavePolicy

`DiffWavePolicy` 是扩散策略网络，包含：

```text
FeatureEncoder
cond_proj: 200 -> 128
DiffWaveModel
DiffusionScheduler
value_head: 200 -> 1
```

其中 `value_head` 不是 PPO critic，不参与 PPO ratio/clip，只作为回报估计辅助 diffusion 训练。

### 4.3 DiffWaveModel

`DiffWaveModel` 是动作扩散模型。它接收噪声动作 `x_t`、扩散步 `t` 和条件特征 `cond`，预测噪声 `pred_noise`。

结构：

```text
input_projection: Conv1d(1 -> residual channels)
diffusion embedding: sinusoidal embedding + MLP
12 个 DiffWaveResidualBlock
skip connection 聚合
output_projection: Conv1d(res_channels -> 1)
```

每个 residual block 包含：

```text
dilated Conv1d
diffusion step projection
condition projection
gated activation: tanh * sigmoid
residual output + skip output
```

### 4.4 ActionValueCritic

`ActionValueCritic` 用于估计当前条件特征下某个动作的价值：

```text
input = [cond_features, action]
MLP: 201 -> 128 -> 128 -> 1
output = Q(s, a)
```

它的作用是给扩散策略提供动作方向指导：

1. 训练时用 discounted return 拟合 `Q(s, a)`。
2. 采样动作时，当 critic 已经训练到一定程度，会从多个 diffusion 候选动作里选择 Q 值最高的动作。
3. 策略训练时加入 `q_guidance_loss = -Q(s, sampled_action)`，鼓励策略生成高 Q 动作。

## 5. 动作生成机制

`DiffWaveAgent.choose_action()` 的逻辑：

```text
1. 编码当前 image / robot_state / graph_state / safety_features
2. value_head 输出 value estimate
3. diffusion 独立采样 8 个候选动作
4. 如果 critic 已训练足够：
       用 ActionValueCritic 评估 8 个候选动作
       选择 Q 值最高的动作
   否则：
       使用第一个 diffusion 动作
       训练模式下加少量高斯探索噪声
5. 动作 clip 到 [-1, 1]
```

Q 引导启用条件：

```text
critic_updates >= 3
success_replay + elite_replay >= 32
q_candidate_count > 1
```

这可以避免训练早期 critic 不准时错误引导动作。

## 6. 损失函数

每个 episode 结束后调用 `learn()`。训练目标由三部分组成：

```text
loss = diffusion_loss
     + value_coef * value_loss
     + q_coef * q_guidance_loss
```

### 6.1 diffusion_loss

对真实执行过的动作加噪，模型预测噪声：

```text
x_t = q_sample(action, t, noise)
pred_noise = diffusion(x_t, t, cond)
diffusion_loss = weighted_mse(pred_noise, noise)
```

权重来自 advantage、成功标记和安全惩罚：

```text
advantage = discounted_return - value_estimate
weight = exp(normalized_advantage / temperature)
成功 episode 提高权重
失败 episode 降低权重
安全惩罚大的动作降低权重
```

### 6.2 value_loss

value head 预测 discounted return：

```text
value_loss = smooth_l1_loss(value_pred, discounted_return)
```

使用 `smooth_l1_loss` 是为了避免成功奖励很大时 value loss 过大，压制 diffusion loss。

### 6.3 q_loss

critic 拟合执行动作的回报：

```text
q_loss = smooth_l1_loss(Q(cond, action), discounted_return)
```

critic 单独用 `critic_optimizer` 更新。

### 6.4 q_guidance_loss

冻结 critic，用当前 policy 采样动作，并最大化 critic 认为好的动作：

```text
q_guidance_loss = -mean(Q(cond, sampled_action))
```

该项让 diffusion policy 不只是模仿已有动作，还能被 critic 引导到更高回报方向。

## 7. Replay 设计

当前 DiffWave 有三类经验：

```text
current episode memory 当前 episode 临时缓存
success_replay         成功 episode 的长期缓存
elite_replay           高回报 episode 的长期缓存
```

`clear_memory()` 只清空当前 episode 的临时缓存，不清空 `success_replay` 和 `elite_replay`。

成功轨迹保留策略：

```text
只要 episode 成功，该 episode 所有 transition 都进入 success_replay。
```

高回报轨迹保留策略：

```text
如果 episode 成功，进入 elite_replay。
如果 episode_return >= 最近 100 个 episode 回报的 70% 分位数，也进入 elite_replay。
```

每次 learn 的每个 epoch 都重新采样 replay，而不是固定一批 replay 重复训练。默认 batch 大小为 128，replay 预算优先分配：

```text
60% success_replay
30% elite_replay
剩余部分优先补 success，其次补 elite
```

这样做的目的：

1. 成功样本不会被单次训练后遗忘。
2. 失败样本仍参与训练，但不会压过稀有成功样本。
3. 高回报失败轨迹也可能保留，用于学习接近成功的动作模式。

## 8. Checkpoint 内容

抓取阶段 checkpoint 保存：

```text
diffwave_catch joint-action policy / optimizer
diffwave_catch joint-action critic / critic optimizer
diffwave_catch joint-action replay state
catch_action_dim = 2
episode 信息
测试成功率信息
```

抬腿阶段 checkpoint 保存：

```text
leg_upper policy / optimizer / critic / critic optimizer / replay
leg_lower policy / optimizer / critic / critic optimizer / replay
ankle policy / optimizer / critic / critic optimizer / replay
```

replay 只保存轻量状态：

```text
最近 64 条 success_replay
最近 64 条 elite_replay
recent_episode_returns
critic_updates
```

旧 checkpoint 没有 replay 字段也可以加载，只是 replay 从空开始。

## 9. 日志指标

当前日志会记录：

```text
episode_num
return_all
goal
loss
diffusion_loss
value_loss
q_loss
q_guidance_loss
success_replay_size
elite_replay_size
q_guided_used
q_guided_action_delta
critic_updates
safety_penalty
rolling_success_rate_100
rolling_mean_return_100
test_grasp_success_rate
```

重点观察：

```text
success_replay_size 是否持续积累成功样本
elite_replay_size 是否持续积累高质量样本
critic_updates 是否持续增加
q_guided_used 是否在训练一段时间后变成 1
rolling_success_rate_100 是否上升
return_all 是否整体上升
```

## 10. 与 PPO 的区别

当前 DiffWave 不使用：

```text
log_prob
PPO ratio
clip_ratio
GAE
entropy_coef
```

PPO 的策略更新是直接提高高 advantage 动作概率、降低低 advantage 动作概率；当前 DiffWave 的策略更新是：

```text
1. 奖励加权 imitation：高回报动作 diffusion loss 权重大。
2. critic/Q guidance：用 Q(s,a) 选择和鼓励更高价值动作。
3. success/elite replay：让成功动作长期反复参与训练。
```

因此这套 DiffWave 更像是：

```text
奖励加权扩散策略
+ 成功轨迹回放
+ 高回报轨迹回放
+ Q critic 动作指导
+ Webots 在线交互训练
```

## 11. 当前训练效果判断建议

训练时不要只看单个 episode 的 `goal`，建议同时看：

```text
1. 最近 100 episode 成功率 rolling_success_rate_100
2. 最近 100 episode 平均回报 rolling_mean_return_100
3. success_replay_size 是否增加
4. q_loss 是否逐渐稳定
5. q_guided_used 是否在 critic warmup 后启用
6. checkpoint 测试成功率 test_grasp_success_rate
```

如果出现下面情况，说明训练链路开始有效：

```text
success_replay_size > 0
elite_replay_size 持续增长
critic_updates 持续增长
q_guided_used 后期变为 1
return_all 的高分 episode 变多
rolling_success_rate_100 有上升趋势
```

如果长时间没有成功样本，DiffWave 会缺少可模仿的高质量动作，这时需要优先调整奖励、探索噪声、动作安全过滤或初始姿态，而不是继续堆网络层数。
