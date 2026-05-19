import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data

from python_scripts.Project_config import device


class FeatureEncoder(nn.Module):
    def __init__(self, node_num, safety_dim=14):
        super().__init__()
        self.node_num = node_num
        self.safety_dim = safety_dim
        self.image_dim = 256
        self.state_dim = 128
        self.graph_hidden_dim = 256
        self.graph_dim = 128
        self.safety_out_dim = 128

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(2, 2), padding=1)
        self.image_pool = nn.AdaptiveAvgPool2d((14, 14))

        self.image_encoder = nn.Sequential(
            nn.Linear(in_features=6272, out_features=1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(in_features=1024, out_features=self.image_dim),
            nn.ReLU(),
            nn.LayerNorm(self.image_dim),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(in_features=20, out_features=self.state_dim),
            nn.ReLU(),
            nn.LayerNorm(self.state_dim),
        )

        self.conv_graph1 = torch_geometric.nn.GraphSAGE(1, self.graph_hidden_dim, 2, aggr="add")
        self.conv_graph2 = torch_geometric.nn.GATConv(self.graph_hidden_dim, self.graph_hidden_dim, aggr="add")
        self.conv_graph3 = torch_geometric.nn.GraphSAGE(self.graph_hidden_dim, self.graph_hidden_dim, 2, aggr="add")
        self.fc_graph = nn.Sequential(
            nn.Linear(self.graph_hidden_dim, self.graph_dim),
            nn.ReLU(),
            nn.LayerNorm(self.graph_dim),
        )

        self.safety_proj = nn.Sequential(
            nn.Linear(self.safety_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, self.safety_out_dim),
            nn.ReLU(),
            nn.LayerNorm(self.safety_out_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.image_dim + self.state_dim + self.graph_dim + self.safety_out_dim, 200),
            nn.ReLU(),
            nn.LayerNorm(200),
        )

    def create_edge_index(self):
        ans = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
             17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ]
        return torch.tensor(ans, dtype=torch.long)

    def _fixed_vector(self, values, size):
        if isinstance(values, torch.Tensor):
            tensor = values.detach().flatten().float()
        else:
            tensor = torch.as_tensor(values, dtype=torch.float32).flatten()
        if tensor.numel() < size:
            tensor = F.pad(tensor, (0, size - tensor.numel()))
        elif tensor.numel() > size:
            tensor = tensor[:size]
        return tensor

    def creat_x(self, x_graph):
        x_graph = self._fixed_vector(x_graph, self.node_num)
        return [[float(x_graph[i].item())] for i in range(self.node_num)]

    def creat_graph(self, x_graph):
        x = torch.as_tensor(self.creat_x(x_graph), dtype=torch.float32)
        edge_index = torch.as_tensor(self.create_edge_index(), dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)
        graph.x = graph.x.to(device)
        graph.edge_index = graph.edge_index.to(device)
        return graph

    def _format_safety_features(self, safety_features):
        if safety_features is None:
            return torch.zeros(1, self.safety_dim, dtype=torch.float32, device=device)

        safety_features = torch.as_tensor(safety_features, dtype=torch.float32, device=device).flatten()
        if safety_features.numel() < self.safety_dim:
            safety_features = F.pad(safety_features, (0, self.safety_dim - safety_features.numel()))
        elif safety_features.numel() > self.safety_dim:
            safety_features = safety_features[:self.safety_dim]
        return safety_features.view(1, self.safety_dim)

    def forward(self, x, state, x_graph, safety_features=None):
        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.image_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.flatten(x)
        image_features = self.image_encoder(x)

        state = self._fixed_vector(state, 20).to(device)
        state_features = self.state_encoder(state)

        graph = self.creat_graph(x_graph)
        edge_index = graph.edge_index
        graph_x = self.conv_graph1(graph.x, edge_index)
        graph_x = self.relu(graph_x)
        graph_x = self.conv_graph2(graph_x, edge_index)
        graph_x = self.relu(graph_x)
        graph_x = self.conv_graph3(graph_x, edge_index)
        graph_x = self.relu(graph_x)
        graph_x = torch.mean(graph_x, dim=0)
        graph_features = self.fc_graph(graph_x)

        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)
        if state_features.dim() == 1:
            state_features = state_features.unsqueeze(0)
        if graph_features.dim() == 1:
            graph_features = graph_features.unsqueeze(0)
        safety_features = self.safety_proj(self._format_safety_features(safety_features))
        features = self.fusion(torch.cat((image_features, state_features, graph_features, safety_features), dim=-1))
        return features


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half_dim = self.dim // 2
        if half_dim == 0:
            return torch.zeros((t.shape[0], self.dim), device=t.device)

        scale = torch.log(torch.tensor(10000.0, device=t.device))
        scale = scale / (half_dim - 1 if half_dim > 1 else 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=t.device)], dim=1)
        return emb


class WaveGradNoiseSchedule(nn.Module):
    # 用连续 sigma 代替离散扩散步，训练时随机采样噪声等级，采样时按高到低逐级去噪。
    def __init__(self, num_steps, sigma_min=0.01, sigma_max=1.0):
        super().__init__()
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        sigmas = torch.exp(
            torch.linspace(
                math.log(self.sigma_max),
                math.log(self.sigma_min),
                int(num_steps),
                dtype=torch.float32,
            )
        )
        self.register_buffer("sigmas", sigmas)

    @property
    def num_steps(self):
        return int(self.sigmas.shape[0])

    def sample(self, batch_size, target_device):
        # log-uniform 采样让模型同时看到高噪声探索和低噪声精修两类场景。
        log_min = math.log(self.sigma_min)
        log_max = math.log(self.sigma_max)
        uniform = torch.rand(int(batch_size), device=target_device)
        sigma = torch.exp(log_min + uniform * (log_max - log_min))
        return sigma.view(-1, 1, 1)

    def sampling_levels(self, steps=None):
        if steps is None or int(steps) >= self.num_steps:
            return self.sigmas
        indices = torch.linspace(0, self.num_steps - 1, int(steps), device=self.sigmas.device)
        return self.sigmas[indices.round().long()]


class WaveGradResidualBlock(nn.Module):
    # 用 cond 和噪声等级生成 FiLM 调制参数，在低维动作空间里做轻量条件去噪。
    def __init__(self, hidden_dim, cond_dim, noise_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.cond_projection = nn.Linear(cond_dim, 2 * hidden_dim)
        self.noise_projection = nn.Linear(noise_dim, 2 * hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, cond, noise_emb):
        scale_shift = self.cond_projection(cond) + self.noise_projection(noise_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        y = self.norm(x)
        y = y * (1.0 + torch.tanh(scale)) + shift
        return x + self.net(y) * 0.70710678


class WaveGradModel(nn.Module):
    # 当前动作维度很低，用残差 MLP 比音频版大卷积结构更贴合机器人动作输出。
    def __init__(
        self,
        action_dim,
        hidden_dim=128,
        cond_dim=128,
        noise_embed_dim=128,
        num_layers=5,
    ):
        super().__init__()
        self.action_dim = max(1, int(action_dim))
        self.input_projection = nn.Linear(self.action_dim, hidden_dim)
        self.noise_embedding = SinusoidalEmbedding(noise_embed_dim)
        self.noise_mlp = nn.Sequential(
            nn.Linear(noise_embed_dim, noise_embed_dim),
            nn.SiLU(),
            nn.Linear(noise_embed_dim, noise_embed_dim),
        )
        self.residual_layers = nn.ModuleList(
            [
                WaveGradResidualBlock(
                    hidden_dim=hidden_dim,
                    cond_dim=cond_dim,
                    noise_dim=noise_embed_dim,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )

    def forward(self, x, sigma, cond):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = x.view(x.shape[0], -1)
        if x.shape[1] < self.action_dim:
            x = F.pad(x, (0, self.action_dim - x.shape[1]))
        elif x.shape[1] > self.action_dim:
            x = x[:, : self.action_dim]

        sigma = sigma.view(sigma.shape[0], -1)[:, 0].clamp(min=1e-4)
        noise_level = torch.log(sigma)
        noise_emb = self.noise_mlp(self.noise_embedding(noise_level))

        h = self.input_projection(x)
        for layer in self.residual_layers:
            h = layer(h, cond, noise_emb)
        return self.output_projection(h).view(-1, 1, self.action_dim)


class WaveGradPolicy(nn.Module):
    def __init__(
        self,
        node_num,
        cond_dim=200,
        model_dim=128,
        res_channels=128,
        num_layers=5,
        dilation_cycle=4,
        diffusion_steps=24,
        safety_dim=14,
        action_dim=1,
    ):
        super().__init__()
        self.action_dim = max(1, int(action_dim))
        self.encoder = FeatureEncoder(node_num, safety_dim=safety_dim)
        self.cond_proj = nn.Linear(cond_dim, model_dim)
        self.diffusion = WaveGradModel(
            action_dim=self.action_dim,
            hidden_dim=res_channels,
            cond_dim=model_dim,
            noise_embed_dim=model_dim,
            num_layers=num_layers,
        )
        self.scheduler = WaveGradNoiseSchedule(diffusion_steps)
        self.value_head = nn.Linear(cond_dim, 1)

    def encode(self, x, state, x_graph, safety_features=None):
        features = self.encoder(x, state, x_graph, safety_features=safety_features)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return features

    def value(self, features):
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return self.value_head(features)

    def diffusion_loss(self, action_seq, cond_features, weights=None):
        if cond_features.dim() == 1:
            cond_features = cond_features.unsqueeze(0)
        cond = self.cond_proj(cond_features)
        action_seq = self._format_action_tensor(action_seq, cond.shape[0], cond.device)
        if weights is None:
            weights = torch.ones_like(action_seq)
        else:
            weights = torch.as_tensor(weights, dtype=torch.float32, device=action_seq.device)
            weights = weights.view(action_seq.shape[0], 1, -1)
            if weights.shape[-1] == 1:
                weights = weights.expand_as(action_seq)
            elif weights.shape[-1] != self.action_dim:
                weights = weights[:, :, :1].expand_as(action_seq)

        # 训练目标是给真实执行动作加连续强度噪声，再让模型预测这部分噪声。
        sigma = self.scheduler.sample(action_seq.shape[0], action_seq.device)
        noise = torch.randn_like(action_seq)
        noisy_action = action_seq + sigma * noise
        pred_noise = self.diffusion(noisy_action, sigma, cond)
        loss = (pred_noise - noise) ** 2
        sigma_weight = (1.0 / (sigma + 0.05)).clamp(0.5, 5.0)
        return (loss * weights * sigma_weight).mean()

    @torch.no_grad()
    def sample_actions(self, cond_features, action_count=None, deterministic_seed=None):
        if cond_features.dim() == 1:
            cond_features = cond_features.unsqueeze(0)
        cond = self.cond_proj(cond_features)
        batch = cond.shape[0]
        action_count = self.action_dim if action_count is None else max(1, int(action_count))

        restore_rng = deterministic_seed is not None
        if restore_rng:
            seed = int(deterministic_seed) % (2 ** 31)
            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        try:
            x = self._sample_from_noise(cond, action_count, steps=self.scheduler.num_steps, detach_noise=True)
            return torch.tanh(x).squeeze(1)
        finally:
            if restore_rng:
                torch.random.set_rng_state(cpu_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)

    def sample_actions_with_grad(self, cond_features, action_count=None, guidance_steps=6):
        if cond_features.dim() == 1:
            cond_features = cond_features.unsqueeze(0)
        cond = self.cond_proj(cond_features)
        action_count = self.action_dim if action_count is None else max(1, int(action_count))
        x = self._sample_from_noise(cond, action_count, steps=guidance_steps, detach_noise=False)
        return torch.tanh(x).squeeze(1)

    def _format_action_tensor(self, actions, batch_size, target_device):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=target_device)
        actions = actions.view(int(batch_size), -1)
        if actions.shape[1] < self.action_dim:
            actions = F.pad(actions, (0, self.action_dim - actions.shape[1]))
        elif actions.shape[1] > self.action_dim:
            actions = actions[:, : self.action_dim]
        return actions.view(int(batch_size), 1, self.action_dim)

    def _sample_from_noise(self, cond, action_count, steps=None, detach_noise=True):
        batch = cond.shape[0]
        action_count = max(1, int(action_count))
        levels = self.scheduler.sampling_levels(steps=steps).to(cond.device)
        x = torch.randn(batch, 1, self.action_dim, device=cond.device) * levels[0].view(1, 1, 1)
        # 从高噪声动作开始，逐级估计 clean_action，并过渡到下一个更低 sigma。
        for idx, sigma_value in enumerate(levels):
            sigma = torch.full((batch, 1, 1), float(sigma_value.item()), device=cond.device)
            pred_noise = self.diffusion(x, sigma, cond)
            clean_action = x - sigma * pred_noise
            if idx + 1 < len(levels):
                next_sigma = levels[idx + 1].view(1, 1, 1).to(cond.device)
                x = clean_action + next_sigma * pred_noise
            else:
                x = clean_action
            x = x.clamp(-3.0, 3.0)

        if detach_noise:
            x = x.detach()
        if action_count < self.action_dim:
            x = x[:, :, :action_count]
        elif action_count > self.action_dim:
            x = F.pad(x, (0, action_count - self.action_dim))
        return x


class ActionValueCritic(nn.Module):
    def __init__(self, cond_dim=200, hidden_dim=128, action_dim=1):
        super().__init__()
        self.action_dim = max(1, int(action_dim))
        self.net = nn.Sequential(
            nn.Linear(cond_dim + self.action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cond_features, actions):
        if cond_features.dim() == 1:
            cond_features = cond_features.unsqueeze(0)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=cond_features.device)
        actions = actions.view(cond_features.shape[0], -1)
        if actions.shape[1] < self.action_dim:
            actions = F.pad(actions, (0, self.action_dim - actions.shape[1]))
        elif actions.shape[1] > self.action_dim:
            actions = actions[:, : self.action_dim]
        x = torch.cat((cond_features, actions), dim=-1)
        return self.net(x).squeeze(-1)


class WaveGradAgent:
    def __init__(
        self,
        node_num,
        env_information=None,
        trajectory_len=22,
        safety_dim=14,
        update_epochs=3,
        max_grad_norm=1.0,
        success_replay_size=600,
        elite_replay_size=1000,
        replay_batch_size=64,
        advantage_temperature=2.0,
        q_coef=0.05,
        q_guidance_batch=8,
        q_guidance_steps=4,
        q_candidate_count=4,
        q_guidance_loss_clip=20.0,
        q_guidance_min_critic_updates=1000,
        q_guidance_min_replay=200,
        replay_action_clip=0.85,
        min_policy_lr=1e-5,
        action_dim=1,
    ):
        self.node_num = node_num
        self.env_information = env_information
        self.trajectory_len = trajectory_len
        self.safety_dim = safety_dim
        self.action_dim = max(1, int(action_dim))

        self.gamma = 0.99
        self.value_coef = 0.5
        self.max_grad_norm = max_grad_norm
        self.lr = 2e-4
        self.lr_decay = 0.9995
        self.min_policy_lr = float(min_policy_lr)
        self.replay_action_clip = float(replay_action_clip)
        self.update_epochs = update_epochs
        self.success_weight = 2.0
        self.failure_weight = 0.75
        self.safety_weight_scale = 0.15
        self.replay_batch_size = replay_batch_size
        self.advantage_temperature = advantage_temperature
        self.q_coef = q_coef
        self.q_guidance_batch = q_guidance_batch
        self.q_guidance_steps = q_guidance_steps
        self.q_candidate_count = max(1, int(q_candidate_count))
        self.q_guidance_loss_clip = float(q_guidance_loss_clip)
        self.q_guidance_min_critic_updates = max(0, int(q_guidance_min_critic_updates))
        self.q_guidance_min_replay = max(0, int(q_guidance_min_replay))

        self.policy = WaveGradPolicy(
            node_num=self.node_num,
            diffusion_steps=24,
            safety_dim=self.safety_dim,
            action_dim=self.action_dim,
        ).to(device)
        self.critic = ActionValueCritic(action_dim=self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        self._clamp_policy_lr()
        self.success_replay_capacity = max(0, int(success_replay_size))
        self.success_replay = []
        self.elite_replay = deque(maxlen=elite_replay_size)
        self.recent_episode_returns = deque(maxlen=100)
        self.critic_updates = 0
        self.q_guided_action_count = 0
        self.action_count_since_learn = 0
        self.q_guided_delta_sum = 0.0
        self.last_action_info = self._empty_action_info()
        self.last_loss_info = self._empty_loss_info()
        self.clear_memory()

    def choose_action(
        self,
        obs,
        x_graph,
        safety_features=None,
        explore=True,
        explore_noise_std=None,
        q_guidance_probability=1.0,
        action_clip=1.0,
        candidate_count=None,
        deterministic_seed=None,
    ):
        if isinstance(obs, (list, tuple)) and len(obs) >= 2:
            x = obs[0]
            state = obs[1]
        else:
            x = obs
            state = x_graph

        if isinstance(x, torch.Tensor):
            x = x.to(device)

        with torch.no_grad():
            features = self.policy.encode(x, state, x_graph, safety_features=safety_features)
            value = self.policy.value(features)
            active_candidate_count = self.q_candidate_count if candidate_count is None else int(candidate_count)
            active_candidate_count = max(1, active_candidate_count)
            use_q_guidance = self._should_use_q_guidance(active_candidate_count, q_guidance_probability)
            sampling_candidate_count = active_candidate_count if use_q_guidance else 1
            candidate_features = features.expand(sampling_candidate_count, -1)
            candidates = self.policy.sample_actions(
                candidate_features,
                action_count=self.action_dim,
                deterministic_seed=deterministic_seed,
            ).view(-1, self.action_dim)
            base_action = candidates[0].detach().cpu().numpy().astype(np.float32)
            action = base_action.copy()
            q_guided_used = False
            q_guided_delta = 0.0

            if use_q_guidance:
                q_values = self.critic(candidate_features, candidates)
                best_idx = int(torch.argmax(q_values).item())
                action = candidates[best_idx].detach().cpu().numpy().astype(np.float32)
                q_guided_used = True
                q_guided_delta = float(np.linalg.norm(action - base_action))
            elif explore:
                noise_std = 0.04 if explore_noise_std is None else float(explore_noise_std)
                if noise_std > 0.0:
                    action = action + np.random.normal(0.0, noise_std, size=self.action_dim).astype(np.float32)

        self._record_action_info(q_guided_used, q_guided_delta)
        clip_value = max(0.01, min(1.0, float(action_clip)))
        clipped_action = np.clip(action, -clip_value, clip_value).astype(np.float32)
        if self.action_dim == 1:
            return float(clipped_action[0]), float(value.squeeze().item())
        return clipped_action.tolist(), float(value.squeeze().item())

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        value,
        safety_features=None,
        success_flag=False,
        safety_penalty=0.0,
    ):
        self.states.append(self._compact_state(state))
        self.actions.append(self._compact_action(action))
        self.rewards.append(float(reward))
        self.next_states.append(self._compact_state(next_state))
        self.values.append(float(value))
        self.dones.append(int(done))
        self.safety_features.append(self._compact_safety_features(safety_features))
        self.success_flags.append(1.0 if success_flag else 0.0)
        self.safety_penalties.append(float(safety_penalty))

    def store_transition_catch(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        value,
        safety_features=None,
        success_flag=False,
        safety_penalty=0.0,
    ):
        self.store_transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            value=value,
            safety_features=safety_features,
            success_flag=success_flag,
            safety_penalty=safety_penalty,
        )

    def _discounted_returns_from(self, rewards, dones):
        if not rewards:
            return np.array([], dtype=np.float32)

        returns = np.zeros(len(rewards), dtype=np.float32)
        running_return = 0.0
        for idx in reversed(range(len(rewards))):
            if dones[idx]:
                running_return = 0.0
            running_return = rewards[idx] + self.gamma * running_return
            returns[idx] = running_return
        return returns

    def _discounted_returns(self):
        return self._discounted_returns_from(self.rewards, self.dones)

    def _training_weights(self, returns, values=None, success_flags=None, safety_penalties=None):
        if returns.size == 0:
            return returns

        if values is None:
            values = np.asarray(self.values, dtype=np.float32)
        else:
            values = np.asarray(values, dtype=np.float32)

        if values.size == returns.size:
            advantage = returns - values
        else:
            advantage = returns

        if advantage.size == 1 or float(np.std(advantage)) < 1e-6:
            weights = np.ones_like(returns, dtype=np.float32)
        else:
            normalized = (advantage - float(np.mean(advantage))) / (float(np.std(advantage)) + 1e-6)
            weights = np.exp(normalized / max(self.advantage_temperature, 1e-6)).astype(np.float32)

        if success_flags is None:
            success_flags = np.asarray(self.success_flags, dtype=np.float32)
        else:
            success_flags = np.asarray(success_flags, dtype=np.float32)
        if safety_penalties is None:
            safety_penalties = np.asarray(self.safety_penalties, dtype=np.float32)
        else:
            safety_penalties = np.asarray(safety_penalties, dtype=np.float32)

        if success_flags.size == weights.size and np.max(success_flags) > 0:
            weights = weights * (1.0 + self.success_weight * success_flags)
            weights = weights * np.where(success_flags > 0.0, 1.0, self.failure_weight)
        elif success_flags.size == weights.size:
            weights = weights * self.failure_weight

        if safety_penalties.size == weights.size:
            weights = weights / (1.0 + self.safety_weight_scale * np.maximum(safety_penalties, 0.0))

        return np.clip(weights, 0.05, 5.0).astype(np.float32)

    def learn(self):
        returns = self._discounted_returns()
        if returns.size == 0:
            self.last_loss_info = self._empty_loss_info()
            return 0.0

        self._clamp_policy_lr()
        current_transitions = self._build_current_episode(returns)
        self._remember_episode(current_transitions)
        total_loss = 0.0
        total_diffusion_loss = 0.0
        total_value_loss = 0.0
        total_q_loss = 0.0
        total_q_guidance_loss = 0.0
        total_q_guidance_loss_used = 0.0
        total_q_guidance_loss_ratio = 0.0
        completed_epochs = 0
        for _ in range(self.update_epochs):
            training_batch = self._make_training_batch(current_transitions)
            if not training_batch:
                continue

            batch_returns = np.asarray([item["return"] for item in training_batch], dtype=np.float32)
            batch_success = np.asarray([item["success_flag"] for item in training_batch], dtype=np.float32)
            batch_safety = np.asarray([item["safety_penalty"] for item in training_batch], dtype=np.float32)
            returns_tensor = torch.tensor(batch_returns, dtype=torch.float32, device=device)
            cond_features, values_pred = self._encode_transitions(training_batch)

            current_values = values_pred.detach().cpu().numpy().astype(np.float32)
            weights = self._training_weights(
                batch_returns,
                values=current_values,
                success_flags=batch_success,
                safety_penalties=batch_safety,
            )
            batch_actions = np.stack([self._action_array(item["action"]) for item in training_batch]).astype(np.float32)
            action_tensor = torch.tensor(
                np.clip(
                    batch_actions,
                    -self.replay_action_clip,
                    self.replay_action_clip,
                ),
                dtype=torch.float32,
                device=device,
            ).view(-1, 1, self.action_dim)
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device).view(-1, 1, 1)

            q_pred = self.critic(cond_features.detach(), action_tensor.view(-1, self.action_dim))
            q_loss = F.smooth_l1_loss(q_pred, returns_tensor)
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            self.critic_updates += 1

            diffusion_loss = self.policy.diffusion_loss(action_tensor, cond_features, weight_tensor)
            value_loss = F.smooth_l1_loss(values_pred, returns_tensor)
            q_guidance_loss = self._q_guidance_loss(cond_features.detach(), weight_tensor.detach())
            q_guidance_loss_used = self._clip_q_guidance_loss(q_guidance_loss)
            q_guidance_contribution = self.q_coef * q_guidance_loss_used
            loss = diffusion_loss + self.value_coef * value_loss + q_guidance_contribution

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            q_guidance_ratio = self._q_guidance_loss_ratio(
                diffusion_loss,
                value_loss,
                q_guidance_contribution,
            )
            total_loss += float(loss.item())
            total_diffusion_loss += float(diffusion_loss.item())
            total_value_loss += float(value_loss.item())
            total_q_loss += float(q_loss.item())
            total_q_guidance_loss += float(q_guidance_loss.item())
            total_q_guidance_loss_used += float(q_guidance_loss_used.item())
            total_q_guidance_loss_ratio += float(q_guidance_ratio)
            completed_epochs += 1

        if completed_epochs == 0:
            self.last_loss_info = self._empty_loss_info()
            self.clear_memory()
            self._reset_action_diagnostics()
            return 0.0

        self.scheduler.step()
        self._clamp_policy_lr()
        avg_loss = total_loss / completed_epochs
        action_count = max(1, self.q_guided_action_count)
        self.last_loss_info = {
            "loss": avg_loss,
            "diffusion_loss": total_diffusion_loss / completed_epochs,
            "value_loss": total_value_loss / completed_epochs,
            "q_loss": total_q_loss / completed_epochs,
            "q_guidance_loss": total_q_guidance_loss / completed_epochs,
            "q_guidance_loss_used": total_q_guidance_loss_used / completed_epochs,
            "q_guidance_loss_ratio": total_q_guidance_loss_ratio / completed_epochs,
            "success_replay_size": len(self.success_replay),
            "elite_replay_size": len(self.elite_replay),
            "q_guided_used": int(self.q_guided_action_count > 0),
            "q_guided_action_delta": self.q_guided_delta_sum / action_count if self.q_guided_action_count > 0 else 0.0,
            "critic_updates": int(self.critic_updates),
            "policy_lr": self.current_policy_lr(),
        }
        self.clear_memory()
        self._reset_action_diagnostics()
        return avg_loss

    def _q_guidance_loss(self, cond_features, weights):
        if self.q_coef <= 0 or cond_features.shape[0] == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=device)

        batch_size = min(self.q_guidance_batch, cond_features.shape[0])
        flat_weights = weights.view(-1)
        if flat_weights.numel() > batch_size:
            _, indices = torch.topk(flat_weights, batch_size)
            guidance_features = cond_features[indices]
        else:
            guidance_features = cond_features

        for param in self.critic.parameters():
            param.requires_grad_(False)
        try:
            guided_actions = self.policy.sample_actions_with_grad(
                guidance_features,
                action_count=self.action_dim,
                guidance_steps=self.q_guidance_steps,
            )
            q_values = self.critic(guidance_features, guided_actions)
            guidance_loss = -q_values.mean()
        finally:
            for param in self.critic.parameters():
                param.requires_grad_(True)
        return guidance_loss

    def _clip_q_guidance_loss(self, q_guidance_loss):
        clip_value = float(self.q_guidance_loss_clip)
        if clip_value <= 0.0:
            return q_guidance_loss
        return torch.clamp(q_guidance_loss, min=-clip_value, max=clip_value)

    def _q_guidance_loss_ratio(self, diffusion_loss, value_loss, q_guidance_contribution):
        with torch.no_grad():
            diffusion_part = torch.abs(diffusion_loss.detach())
            value_part = torch.abs(value_loss.detach() * self.value_coef)
            q_part = torch.abs(q_guidance_contribution.detach())
            denominator = diffusion_part + value_part + q_part + 1e-8
            return float((q_part / denominator).item())

    def _encode_transitions(self, transitions):
        features = []
        values = []
        for item in transitions:
            state = item["state"]
            safety_features = item.get("safety_features")
            encoded = self.policy.encode(
                state[0],
                state[1],
                state[2],
                safety_features=safety_features,
            )
            features.append(encoded)
            values.append(self.policy.value(encoded))
        return torch.cat(features, dim=0), torch.cat(values).squeeze(-1)

    def _build_current_episode(self, returns):
        episode_success = 1.0 if any(flag > 0.0 for flag in self.success_flags) else 0.0
        transitions = []
        for idx in range(len(self.actions)):
            transitions.append(
                {
                    "state": self.states[idx],
                    "action": self._transition_action(self.actions[idx]),
                    "reward": float(self.rewards[idx]),
                    "next_state": self.next_states[idx],
                    "done": int(self.dones[idx]),
                    "value": float(self.values[idx]),
                    "return": float(returns[idx]),
                    "safety_features": self.safety_features[idx],
                    "success_flag": episode_success,
                    "safety_penalty": float(self.safety_penalties[idx]),
                }
            )
        return transitions

    def _remember_episode(self, transitions):
        if not transitions:
            return

        episode_return = float(sum(item["reward"] for item in transitions))
        episode_success = any(item["success_flag"] > 0.0 for item in transitions)
        elite_threshold = (
            float(np.percentile(np.asarray(self.recent_episode_returns, dtype=np.float32), 90.0))
            if len(self.recent_episode_returns) >= 5
            else 0.0
        )

        if episode_success:
            for item in transitions:
                self._add_success_replay_item(item, episode_return)

        if episode_return >= elite_threshold:
            for item in transitions:
                self.elite_replay.append(dict(item))

        self.recent_episode_returns.append(episode_return)

    def _add_success_replay_item(self, item, episode_return=None):
        if self.success_replay_capacity <= 0:
            return

        replay_item = dict(item)
        if episode_return is None:
            score = self._success_replay_score(replay_item)
        else:
            score = float(episode_return)
        replay_item["episode_return"] = score
        replay_item["success_replay_score"] = score

        if len(self.success_replay) < self.success_replay_capacity:
            self.success_replay.append(replay_item)
            self._sort_success_replay()
            return

        lowest_score = self._success_replay_score(self.success_replay[-1])
        if score > lowest_score:
            self.success_replay[-1] = replay_item
            self._sort_success_replay()

    def _success_replay_score(self, item):
        if not isinstance(item, dict):
            return 0.0
        for key in ("success_replay_score", "episode_return", "return", "reward"):
            try:
                return float(item[key])
            except (KeyError, TypeError, ValueError):
                continue
        return 0.0

    def _sort_success_replay(self):
        self.success_replay.sort(key=self._success_replay_score, reverse=True)

    def _make_training_batch(self, current_transitions):
        batch = list(current_transitions)
        replay_budget = max(0, self.replay_batch_size - len(batch))
        if replay_budget <= 0:
            return batch

        success_count = min(len(self.success_replay), int(replay_budget * 0.6))
        elite_count = min(len(self.elite_replay), int(replay_budget * 0.3))
        remaining = replay_budget - success_count - elite_count
        if remaining > 0 and len(self.success_replay) > success_count:
            success_count += min(remaining, len(self.success_replay) - success_count)
            remaining = replay_budget - success_count - elite_count
        if remaining > 0 and len(self.elite_replay) > elite_count:
            elite_count += min(remaining, len(self.elite_replay) - elite_count)

        batch.extend(self._sample_replay(self.success_replay, success_count))
        batch.extend(self._sample_replay(self.elite_replay, elite_count))
        random.shuffle(batch)
        return batch

    def _sample_replay(self, replay_buffer, count):
        if count <= 0 or not replay_buffer:
            return []
        count = min(count, len(replay_buffer))
        return [dict(item) for item in random.sample(list(replay_buffer), count)]

    def _compact_state(self, state):
        if isinstance(state, (list, tuple)) and len(state) >= 3:
            return [
                self._compact_array(state[0], np.float16),
                self._compact_array(state[1], np.float32),
                self._compact_array(state[2], np.float32),
            ]
        return [
            self._compact_array(state, np.float16),
            np.zeros(20, dtype=np.float32),
            np.zeros(self.node_num, dtype=np.float32),
        ]

    def _compact_array(self, value, dtype):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype).copy()

    def _compact_safety_features(self, safety_features):
        if safety_features is None:
            return None
        if isinstance(safety_features, torch.Tensor):
            safety_features = safety_features.detach().cpu().numpy()
        return np.asarray(safety_features, dtype=np.float32).copy()

    def _compact_action(self, action):
        action_array = self._action_array(action)
        if self.action_dim == 1:
            return float(action_array[0])
        return action_array.astype(np.float32).copy()

    def _action_array(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size < self.action_dim:
            action_array = np.pad(action_array, (0, self.action_dim - action_array.size), mode="constant")
        elif action_array.size > self.action_dim:
            action_array = action_array[: self.action_dim]
        return action_array.astype(np.float32, copy=False)

    def _transition_action(self, action):
        action_array = self._action_array(action)
        if self.action_dim == 1:
            return float(action_array[0])
        return action_array.astype(np.float32).copy()

    def _empty_loss_info(self):
        return {
            "loss": 0.0,
            "diffusion_loss": 0.0,
            "value_loss": 0.0,
            "q_loss": 0.0,
            "q_guidance_loss": 0.0,
            "q_guidance_loss_used": 0.0,
            "q_guidance_loss_ratio": 0.0,
            "success_replay_size": len(self.success_replay) if hasattr(self, "success_replay") else 0,
            "elite_replay_size": len(self.elite_replay) if hasattr(self, "elite_replay") else 0,
            "q_guided_used": 0,
            "q_guided_action_delta": 0.0,
            "critic_updates": int(self.critic_updates) if hasattr(self, "critic_updates") else 0,
            "policy_lr": self.current_policy_lr() if hasattr(self, "optimizer") else 0.0,
        }

    def _clamp_policy_lr(self):
        for group in self.optimizer.param_groups:
            group["lr"] = max(float(group.get("lr", self.lr)), self.min_policy_lr)

    def current_policy_lr(self):
        if not hasattr(self, "optimizer") or not self.optimizer.param_groups:
            return 0.0
        return float(self.optimizer.param_groups[0].get("lr", 0.0))

    def _empty_action_info(self):
        return {
            "q_guided_used": False,
            "q_guided_action_delta": 0.0,
            "critic_updates": int(self.critic_updates) if hasattr(self, "critic_updates") else 0,
        }

    def _record_action_info(self, q_guided_used, q_guided_delta):
        self.action_count_since_learn += 1
        if q_guided_used:
            self.q_guided_action_count += 1
            self.q_guided_delta_sum += float(q_guided_delta)
        self.last_action_info = {
            "q_guided_used": bool(q_guided_used),
            "q_guided_action_delta": float(q_guided_delta),
            "critic_updates": int(self.critic_updates),
        }

    def _reset_action_diagnostics(self):
        self.q_guided_action_count = 0
        self.action_count_since_learn = 0
        self.q_guided_delta_sum = 0.0

    def _should_use_q_guidance(self, candidate_count=None, q_guidance_probability=1.0):
        replay_size = len(self.success_replay) + len(self.elite_replay)
        active_candidate_count = self.q_candidate_count if candidate_count is None else int(candidate_count)
        probability = max(0.0, min(1.0, float(q_guidance_probability)))
        return (
            active_candidate_count > 1
            and probability > 0.0
            and random.random() <= probability
            and self.critic_updates >= self.q_guidance_min_critic_updates
            and replay_size >= self.q_guidance_min_replay
        )

    def export_replay_state(self, max_items=64):
        max_items = max(0, int(max_items))
        return {
            "success_replay": [dict(item) for item in list(self.success_replay)[:max_items]],
            "elite_replay": [dict(item) for item in list(self.elite_replay)[-max_items:]],
            "recent_episode_returns": list(self.recent_episode_returns),
            "critic_updates": int(self.critic_updates),
        }

    def load_replay_state(self, replay_state):
        if not isinstance(replay_state, dict):
            return
        for item in replay_state.get("success_replay", []):
            if isinstance(item, dict):
                self._add_success_replay_item(item)
        for item in replay_state.get("elite_replay", []):
            if isinstance(item, dict):
                self.elite_replay.append(dict(item))
        for value in replay_state.get("recent_episode_returns", []):
            try:
                self.recent_episode_returns.append(float(value))
            except (TypeError, ValueError):
                continue
        try:
            self.critic_updates = max(self.critic_updates, int(replay_state.get("critic_updates", 0)))
        except (TypeError, ValueError):
            pass

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.values = []
        self.dones = []
        self.safety_features = []
        self.success_flags = []
        self.safety_penalties = []


WaveGradCatchAgent = WaveGradAgent
