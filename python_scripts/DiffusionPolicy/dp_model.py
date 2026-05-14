"""Pure Diffusion Policy model and online denoising agent.

This module is intentionally independent from PPO-style actor-critic logic.
The policy generates action chunks with a conditional diffusion model, and the
online update trains only the denoising objective on collected action chunks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv

    _HAS_PYG = True
except Exception:
    SAGEConv = None
    _HAS_PYG = False


def resolve_device(device: Optional[str | torch.device] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _to_numpy(x: Any, dtype=np.float32) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=False)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(x, dtype=dtype)


def _prepare_image_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(x):
        img = x.detach().float()
    else:
        img = torch.as_tensor(x, dtype=torch.float32)

    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.dim() == 3:
        if img.shape[0] in (1, 3) and img.shape[1] > 8 and img.shape[2] > 8:
            img = img.unsqueeze(0)
        elif img.shape[-1] in (1, 3) and img.shape[0] > 8 and img.shape[1] > 8:
            img = img.permute(2, 0, 1).unsqueeze(0)
        else:
            img = img.unsqueeze(1)
    elif img.dim() == 4:
        if img.shape[1] not in (1, 3) and img.shape[-1] in (1, 3):
            img = img.permute(0, 3, 1, 2)
    else:
        raise ValueError("unsupported image shape: %s" % (tuple(img.shape),))

    if img.shape[1] != 1:
        img = img[:, :1, :, :]
    img = img.to(device)
    if img.numel() and img.max() > 1.0:
        img = img / 255.0
    return torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)


def _prepare_state_tensor(x: Any, target_dim: int, device: torch.device) -> torch.Tensor:
    arr = _to_numpy(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    arr = arr.reshape(arr.shape[0], -1)
    out = np.zeros((arr.shape[0], int(target_dim)), dtype=np.float32)
    n = min(arr.shape[1], int(target_dim))
    if n > 0:
        out[:, :n] = arr[:, :n]
    tensor = torch.from_numpy(out).to(device)
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class JointGraphEncoder(nn.Module):
    def __init__(self, node_num: int = 19, hidden_dim: int = 64) -> None:
        super().__init__()
        self.node_num = int(node_num)
        self.hidden_dim = int(hidden_dim)

        if _HAS_PYG:
            self.conv1 = SAGEConv(1, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        else:
            self.fallback = nn.Sequential(
                nn.Linear(node_num, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )

        self.register_buffer("edge_index", self._build_edge_index(), persistent=False)

    def _build_edge_index(self) -> torch.Tensor:
        src = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18,
            1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17,
        ]
        dst = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 17,
            0, 1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        ]
        return torch.tensor([src, dst], dtype=torch.long)

    def forward(self, x_graph: torch.Tensor) -> torch.Tensor:
        if x_graph.dim() == 1:
            x_graph = x_graph.unsqueeze(0)

        if not _HAS_PYG:
            return self.fallback(x_graph[:, : self.node_num])

        outs: List[torch.Tensor] = []
        for b in range(x_graph.size(0)):
            x = x_graph[b, : self.node_num].view(self.node_num, 1)
            h = F.silu(self.conv1(x, self.edge_index))
            h = F.silu(self.conv2(h, self.edge_index))
            h = F.silu(self.conv3(h, self.edge_index))
            outs.append(h.mean(dim=0))
        return torch.stack(outs, dim=0)


class ObservationEncoder(nn.Module):
    def __init__(self, state_dim: int = 20, node_num: int = 19, fused_dim: int = 256) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.node_num = int(node_num)

        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.SiLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.graph_encoder = JointGraphEncoder(node_num=node_num, hidden_dim=64)
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 64, fused_dim),
            nn.SiLU(),
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim),
            nn.SiLU(),
        )

    def forward(self, image: Any, state: Any, graph_state: Any, device: torch.device) -> torch.Tensor:
        img = _prepare_image_tensor(image, device)
        state_tensor = _prepare_state_tensor(state, self.state_dim, device)
        graph_tensor = _prepare_state_tensor(graph_state, self.node_num, device)
        img_feat = self.image_encoder(img)
        state_feat = self.state_encoder(state_tensor)
        graph_feat = self.graph_encoder(graph_tensor)
        if not (img_feat.size(0) == state_feat.size(0) == graph_feat.size(0)):
            raise RuntimeError(
                "encoder batch mismatch: image=%s state=%s graph=%s"
                % (tuple(img_feat.shape), tuple(state_feat.shape), tuple(graph_feat.shape))
            )
        return torch.nan_to_num(
            self.fusion(torch.cat([img_feat, state_feat, graph_feat], dim=-1)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DiffusionDenoiser(nn.Module):
    def __init__(
        self,
        chunk_len: int,
        action_dim: int,
        context_dim: int,
        hidden_dim: int = 512,
        time_dim: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.chunk_len = int(chunk_len)
        self.action_dim = int(action_dim)
        self.time_dim = int(time_dim)
        flat_dim = self.chunk_len * self.action_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_proj = nn.Linear(flat_dim + context_dim + hidden_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim) for _ in range(depth)])
        self.out_proj = nn.Linear(hidden_dim, flat_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size = x_t.size(0)
        time_feat = self.time_mlp(sinusoidal_timestep_embedding(t, self.time_dim))
        flat = x_t.reshape(batch_size, -1)
        h = F.silu(self.in_proj(torch.cat([flat, context, time_feat], dim=-1)))
        out = self.out_proj(self.blocks(h))
        return out.view(batch_size, self.chunk_len, self.action_dim)


class DiffusionPolicyActorCritic(nn.Module):
    """Name kept for checkpoint/import compatibility; this is no longer actor-critic."""

    def __init__(
        self,
        state_dim: int,
        node_num: int,
        action_dim: int,
        chunk_len: int,
        diffusion_steps: int = 20,
        context_dim: int = 256,
        hidden_dim: int = 512,
        action_limit: float = 1.0,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.node_num = int(node_num)
        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.diffusion_steps = int(diffusion_steps)
        self.action_limit = float(action_limit)

        self.encoder = ObservationEncoder(state_dim=state_dim, node_num=node_num, fused_dim=context_dim)
        self.denoiser = DiffusionDenoiser(
            chunk_len=chunk_len,
            action_dim=action_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
        )

        betas = torch.linspace(float(beta_start), float(beta_end), self.diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def encode(self, image: Any, state: Any, graph_state: Any, device: torch.device) -> torch.Tensor:
        return self.encoder(image, state, graph_state, device)

    def _reverse_sample(
        self,
        context: torch.Tensor,
        init_noise: torch.Tensor,
        steps: Optional[int] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        sample_steps = int(steps or self.diffusion_steps)
        sample_steps = max(1, min(sample_steps, self.diffusion_steps))
        timesteps = torch.linspace(
            self.diffusion_steps - 1,
            0,
            sample_steps,
            dtype=torch.long,
            device=init_noise.device,
        )
        x = init_noise
        for i, t in enumerate(timesteps):
            t_batch = torch.full((x.size(0),), int(t.item()), dtype=torch.long, device=x.device)
            alpha_bar_t = self.alpha_bars[t]
            if i + 1 < len(timesteps):
                alpha_bar_prev = self.alpha_bars[timesteps[i + 1]]
            else:
                alpha_bar_prev = torch.tensor(1.0, dtype=x.dtype, device=x.device)

            eps = self.denoiser(x, t_batch, context)
            x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            x0 = x0.clamp(-3.0, 3.0)

            if i == len(timesteps) - 1:
                x = x0
                break

            if eta > 0.0:
                sigma = eta * torch.sqrt(
                    (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                    * (1.0 - alpha_bar_t / alpha_bar_prev)
                )
            else:
                sigma = torch.tensor(0.0, dtype=x.dtype, device=x.device)
            direction = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=0.0)) * eps
            x = torch.sqrt(alpha_bar_prev) * x0 + direction
            if eta > 0.0:
                x = x + sigma * torch.randn_like(x)
        return x

    @torch.no_grad()
    def sample_chunk(
        self,
        image: Any,
        state: Any,
        graph_state: Any,
        device: torch.device,
        deterministic: bool = False,
        init_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        context = self.encode(image, state, graph_state, device)
        batch_size = context.size(0)
        if init_noise is None:
            if deterministic:
                init_noise = torch.zeros(batch_size, self.chunk_len, self.action_dim, device=device)
            else:
                init_noise = torch.randn(batch_size, self.chunk_len, self.action_dim, device=device)
        else:
            init_noise = init_noise.to(device)

        raw_actions = self._reverse_sample(context, init_noise, steps=self.diffusion_steps, eta=0.0)
        actions = torch.tanh(raw_actions) * self.action_limit
        return {
            "init_noise": init_noise,
            "raw_actions": raw_actions,
            "actions": actions,
        }

    def denoising_loss(
        self,
        image: Any,
        state: Any,
        graph_state: Any,
        target_raw_actions: torch.Tensor,
        executed_lens: torch.Tensor,
        sample_weights: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        context = self.encode(image, state, graph_state, device)
        target = target_raw_actions.to(device).float()
        batch_size = target.size(0)
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=device)
        eps = torch.randn_like(target)
        alpha_bar = self.alpha_bars[t].view(batch_size, 1, 1)
        x_t = torch.sqrt(alpha_bar) * target + torch.sqrt(1.0 - alpha_bar) * eps
        eps_pred = self.denoiser(x_t, t, context)

        step_mask = (
            torch.arange(self.chunk_len, device=device).unsqueeze(0)
            < executed_lens.to(device).long().clamp(min=1, max=self.chunk_len).unsqueeze(1)
        ).float()
        mse = F.mse_loss(eps_pred, eps, reduction="none").mean(dim=-1)
        per_sample = (mse * step_mask).sum(dim=1) / step_mask.sum(dim=1).clamp_min(1.0)

        weights = sample_weights.to(device).float().clamp_min(0.0)
        weights = weights / weights.mean().clamp_min(1e-6)
        return (per_sample * weights).mean()


@dataclass
class DPTransition:
    obs_image: np.ndarray
    obs_state: np.ndarray
    obs_graph: np.ndarray
    raw_actions: np.ndarray
    executed_len: int
    reward_sum: float
    done: bool
    success: bool
    sample_weight: float
    info: Dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    def __init__(self, capacity: int = 4096) -> None:
        self.capacity = int(capacity)
        self.transitions: List[DPTransition] = []

    def add(self, transition: DPTransition) -> None:
        self.transitions.append(transition)
        if len(self.transitions) > self.capacity:
            del self.transitions[0 : len(self.transitions) - self.capacity]

    def clear(self) -> None:
        self.transitions.clear()

    def sample_indices(self, batch_size: int) -> np.ndarray:
        n = len(self.transitions)
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        size = min(int(batch_size), n)
        return np.random.choice(n, size=size, replace=False)

    def __len__(self) -> int:
        return len(self.transitions)


@dataclass
class OnlineDPConfig:
    name: str
    action_dim: int
    chunk_len: int = 8
    state_dim: int = 20
    node_num: int = 19
    diffusion_steps: int = 20
    action_limit: float = 1.0
    learning_rate: float = 1e-4
    update_epochs: int = 4
    minibatch_size: int = 32
    max_grad_norm: float = 0.5
    replay_capacity: int = 4096
    device: str = "cuda"
    seed: int = 42


class OnlineDiffusionAgent:
    def __init__(self, cfg: OnlineDPConfig) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        set_global_seed(cfg.seed)
        self.policy = DiffusionPolicyActorCritic(
            state_dim=cfg.state_dim,
            node_num=cfg.node_num,
            action_dim=cfg.action_dim,
            chunk_len=cfg.chunk_len,
            diffusion_steps=cfg.diffusion_steps,
            action_limit=cfg.action_limit,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        self.buffer = ReplayBuffer(capacity=cfg.replay_capacity)
        self.update_count = 0

    @torch.no_grad()
    def act(self, obs_image: Any, obs_state: Any, obs_graph: Any, deterministic: bool = False) -> Dict[str, Any]:
        out = self.policy.sample_chunk(obs_image, obs_state, obs_graph, self.device, deterministic=deterministic)
        return {
            "actions": out["actions"].detach().cpu().numpy()[0],
            "raw_actions": out["raw_actions"].detach().cpu().numpy()[0],
            "init_noise": out["init_noise"].detach().cpu().numpy()[0],
        }

    def store_transition(self, transition: DPTransition) -> None:
        self.buffer.add(transition)

    def clear_buffer(self) -> None:
        self.buffer.clear()

    def _gather_batch(self, indices: Sequence[int]) -> Dict[str, torch.Tensor]:
        items = [self.buffer.transitions[int(i)] for i in indices]
        images = np.stack([t.obs_image for t in items], axis=0).astype(np.float32)
        states = np.stack([t.obs_state for t in items], axis=0).astype(np.float32)
        graphs = np.stack([t.obs_graph for t in items], axis=0).astype(np.float32)
        raw_actions = np.stack([t.raw_actions for t in items], axis=0).astype(np.float32)
        executed_lens = np.asarray([t.executed_len for t in items], dtype=np.int64)
        weights = np.asarray([t.sample_weight for t in items], dtype=np.float32)
        return {
            "images": torch.from_numpy(images),
            "states": torch.from_numpy(states),
            "graphs": torch.from_numpy(graphs),
            "raw_actions": torch.from_numpy(raw_actions),
            "executed_lens": torch.from_numpy(executed_lens),
            "weights": torch.from_numpy(weights),
        }

    def update(self) -> Dict[str, float]:
        if len(self.buffer) == 0:
            return {"updated": 0.0, "diffusion_loss": 0.0, "update_count": float(self.update_count)}

        self.policy.train()
        losses: List[float] = []
        grad_norms: List[float] = []
        num_batches = max(1, math.ceil(len(self.buffer) / max(1, self.cfg.minibatch_size)))

        for _ in range(int(self.cfg.update_epochs)):
            for _ in range(num_batches):
                batch = self._gather_batch(self.buffer.sample_indices(self.cfg.minibatch_size))
                loss = self.policy.denoising_loss(
                    image=batch["images"].to(self.device),
                    state=batch["states"].to(self.device),
                    graph_state=batch["graphs"].to(self.device),
                    target_raw_actions=batch["raw_actions"].to(self.device),
                    executed_lens=batch["executed_lens"].to(self.device),
                    sample_weights=batch["weights"].to(self.device),
                    device=self.device,
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
                grad_norms.append(float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm))

        self.update_count += 1
        return {
            "updated": 1.0,
            "diffusion_loss": float(np.mean(losses)) if losses else 0.0,
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "buffer_size": float(len(self.buffer)),
            "update_count": float(self.update_count),
        }

    def save_checkpoint(
        self,
        save_path: str | os.PathLike[str],
        *,
        episode: int,
        score: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        save_path = str(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": asdict(self.cfg),
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "episode": int(episode),
                "score": float(score),
                "extra": dict(extra or {}),
            },
            save_path,
        )
        return save_path

    def load_checkpoint(self, checkpoint_path: str | os.PathLike[str], strict: bool = False) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "policy" in checkpoint:
            self.policy.load_state_dict(checkpoint["policy"], strict=strict)
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            return checkpoint
        self.policy.load_state_dict(checkpoint, strict=strict)
        return {"episode": 0, "score": 0.0}
