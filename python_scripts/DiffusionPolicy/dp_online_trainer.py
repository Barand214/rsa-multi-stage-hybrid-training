"""External pure Diffusion Policy trainer for the Webots socket environment."""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dp_client import WebotsDPClient
from dp_model import DPTransition, OnlineDPConfig, OnlineDiffusionAgent, resolve_device


@dataclass
class StageRuntimeConfig:
    name: str
    action_dim: int
    chunk_len: int
    max_stage_steps: int
    state_dim: int = 20
    node_num: int = 19
    diffusion_steps: int = 20
    action_limit: float = 1.0
    learning_rate: float = 1e-4
    update_epochs: int = 4
    minibatch_size: int = 32
    max_grad_norm: float = 0.5
    replay_capacity: int = 4096
    update_every_episodes: int = 10
    min_buffer_transitions: int = 32
    checkpoint_every_episodes: int = 200
    checkpoint_dir: str = ""
    seed: int = 42

    def to_agent_cfg(self, device: str) -> OnlineDPConfig:
        return OnlineDPConfig(
            name=self.name,
            action_dim=self.action_dim,
            chunk_len=self.chunk_len,
            state_dim=self.state_dim,
            node_num=self.node_num,
            diffusion_steps=self.diffusion_steps,
            action_limit=self.action_limit,
            learning_rate=self.learning_rate,
            update_epochs=self.update_epochs,
            minibatch_size=self.minibatch_size,
            max_grad_norm=self.max_grad_norm,
            replay_capacity=self.replay_capacity,
            device=device,
            seed=self.seed,
        )


@dataclass
class TrainerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    device: str = field(default_factory=lambda: str(resolve_device()))
    total_episodes: int = 1000
    eval_episodes: int = 20
    exp_name: str = "pure_diffusion_policy"
    save_dir: str = "python_scripts/DiffusionPolicy/checkpoint"
    close_server_on_exit: bool = True
    seed: int = 42
    grasp_trigger_step: int = 19
    grasp_goal: Optional[Tuple[float, float]] = None
    tai_goal: Optional[Tuple[float, float]] = None
    test_interval: int = 100
    save_interval: int = 200
    num_test_episodes: int = 50
    ranking_top_n: int = 5
    ranking_metric: str = "grasp"
    grasp: StageRuntimeConfig = field(default_factory=lambda: StageRuntimeConfig(
        name="grasp",
        action_dim=2,
        chunk_len=8,
        max_stage_steps=21,
        min_buffer_transitions=32,
        update_every_episodes=10,
        checkpoint_every_episodes=200,
        seed=42,
    ))
    tai: StageRuntimeConfig = field(default_factory=lambda: StageRuntimeConfig(
        name="tai",
        action_dim=3,
        chunk_len=8,
        max_stage_steps=21,
        min_buffer_transitions=16,
        update_every_episodes=5,
        checkpoint_every_episodes=200,
        seed=84,
    ))

    def __post_init__(self) -> None:
        root = Path(self.save_dir)
        root.mkdir(parents=True, exist_ok=True)
        if not self.grasp.checkpoint_dir:
            self.grasp.checkpoint_dir = str(root / "checkpoints" / "grasp")
        if not self.tai.checkpoint_dir:
            self.tai.checkpoint_dir = str(root / "checkpoints" / "tai")
        Path(self.grasp.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.tai.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def make_reset_options(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "grasp_trigger_step": int(self.grasp_trigger_step),
            "max_grasp_steps": int(self.grasp.max_stage_steps),
            "max_tai_steps": int(self.tai.max_stage_steps),
        }
        if self.grasp_goal is not None:
            options["grasp_goal"] = list(self.grasp_goal)
        if self.tai_goal is not None:
            options["tai_goal"] = list(self.tai_goal)
        return options


@dataclass
class StageEpisodeSummary:
    stage: str
    success: bool
    invalid_abort: bool
    steps: int
    decisions: int
    episode_return: float
    info: Dict[str, Any] = field(default_factory=dict)


class MovingAverageTracker:
    def __init__(self, window: int = 50) -> None:
        self.window = int(window)
        self.values: List[float] = []

    def add(self, value: float) -> float:
        self.values.append(float(value))
        if len(self.values) > self.window:
            self.values.pop(0)
        return self.mean

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0


class ModelRanking:
    def __init__(self, top_n: int = 5) -> None:
        self.top_n = int(top_n)
        self.rankings: List[Tuple[float, List[str], int]] = []

    def add_and_manage(self, score: float, model_paths: List[str], episode_id: int) -> None:
        model_paths = [str(p) for p in model_paths if p]
        if len(self.rankings) < self.top_n:
            heapq.heappush(self.rankings, (float(score), model_paths, int(episode_id)))
            return
        if score <= self.rankings[0][0]:
            for path in model_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
            return
        _, old_paths, _ = heapq.heappop(self.rankings)
        for path in old_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        heapq.heappush(self.rankings, (float(score), model_paths, int(episode_id)))


def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def _sample_weight(reward_sum: float, success: bool, invalid_abort: bool) -> float:
    if invalid_abort:
        return 0.0
    if success:
        return 8.0
    if reward_sum > 0.0:
        return 1.0 + min(float(reward_sum), 2.0)
    return 0.2


class TwoStageSocketTrainer:
    def __init__(self, cfg: Optional[TrainerConfig] = None) -> None:
        self.cfg = cfg or TrainerConfig()
        self.client = WebotsDPClient(host=self.cfg.host, port=self.cfg.port)
        self.grasp_agent = OnlineDiffusionAgent(self.cfg.grasp.to_agent_cfg(self.cfg.device))
        self.tai_agent = OnlineDiffusionAgent(self.cfg.tai.to_agent_cfg(self.cfg.device))

        self.save_dir = Path(self.cfg.save_dir)
        self.config_path = self.save_dir / "config.json"
        self.metrics_path = self.save_dir / "metrics.jsonl"
        self.summary_path = self.save_dir / "summary.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(_make_json_safe(asdict(self.cfg)), f, ensure_ascii=False, indent=2)

        from python_scripts.PPO_Log_write import Log_write

        self.log_writer_catch = Log_write()
        self.log_writer_tai = Log_write()
        self.catch_log_path = self._make_stage_log_path("catch")
        self.tai_log_path = self._make_stage_log_path("tai")

        self.grasp_episode_count = 0
        self.tai_episode_count = 0
        self.best_test_score = -float("inf")
        self.grasp_success_tracker = MovingAverageTracker(window=50)
        self.tai_success_tracker = MovingAverageTracker(window=50)
        self.pipeline_success_tracker = MovingAverageTracker(window=50)
        self.model_ranking = ModelRanking(top_n=self.cfg.ranking_top_n)

    def _make_stage_log_path(self, stage: str) -> str:
        import glob
        import re

        if stage == "catch":
            log_dir = self.save_dir / "logs" / "catch"
            prefix = "catch_log"
        else:
            log_dir = self.save_dir / "logs" / "tai"
            prefix = "tai_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        latest_num = 0
        for log_path in glob.glob(str(log_dir / ("%s_*.json" % prefix))):
            match = re.search(r"%s_(\d+)\.json" % prefix, str(log_path))
            if match:
                latest_num = max(latest_num, int(match.group(1)))
        return str(log_dir / ("%s_%d.json" % (prefix, latest_num + 1)))

    def close(self) -> None:
        if self.cfg.close_server_on_exit:
            self.client.close()

    def load_stage_checkpoints(self, grasp_checkpoint: Optional[str] = None, tai_checkpoint: Optional[str] = None) -> None:
        if grasp_checkpoint:
            self.grasp_agent.load_checkpoint(grasp_checkpoint, strict=False)
            print("[grasp] loaded checkpoint: %s" % grasp_checkpoint, flush=True)
        if tai_checkpoint:
            self.tai_agent.load_checkpoint(tai_checkpoint, strict=False)
            print("[tai] loaded checkpoint: %s" % tai_checkpoint, flush=True)

    def _maybe_update_agent(
        self,
        agent: OnlineDiffusionAgent,
        stage_cfg: StageRuntimeConfig,
        stage_episode_count: int,
    ) -> Dict[str, float]:
        if len(agent.buffer) < stage_cfg.min_buffer_transitions:
            return {"updated": 0.0, "diffusion_loss": 0.0, "buffer_size": float(len(agent.buffer))}
        if stage_episode_count % stage_cfg.update_every_episodes != 0:
            return {"updated": 0.0, "diffusion_loss": 0.0, "buffer_size": float(len(agent.buffer))}
        return agent.update()

    @staticmethod
    def _clone_obs(obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = np.asarray(obs["image"], dtype=np.float32).copy()
        state = np.asarray(obs["robot_state"], dtype=np.float32).copy()
        graph = np.asarray(obs["graph_state"], dtype=np.float32).copy()
        return image, state, graph

    def run_grasp_stage(
        self,
        episode_idx: int,
        start_obs: Dict[str, Any],
        *,
        training: bool,
        deterministic: bool,
    ) -> Tuple[StageEpisodeSummary, Dict[str, Any]]:
        cfg = self.cfg.grasp
        obs = start_obs
        steps = 0
        decisions = 0
        episode_return = 0.0
        success = False
        invalid_abort = False
        last_info: Dict[str, Any] = {}

        while steps < cfg.max_stage_steps and not success and not invalid_abort:
            obs_image, obs_state, obs_graph = self._clone_obs(obs)
            chunk = self.grasp_agent.act(obs_image, obs_state, obs_graph, deterministic=deterministic)
            actions = np.asarray(chunk["actions"], dtype=np.float32)
            raw_actions = np.asarray(chunk["raw_actions"], dtype=np.float32)

            reward_sum = 0.0
            executed_len = 0
            done_flag = False

            for local_idx in range(cfg.chunk_len):
                reply = self.client.step_grasp(actions[local_idx])
                obs = reply["obs"]
                reward = float(reply["reward"])
                terminated = bool(reply["terminated"])
                truncated = bool(reply["truncated"])
                info = dict(reply["info"])
                last_info = info

                reward_sum += reward
                episode_return += reward
                executed_len += 1
                steps += 1

                self.log_writer_catch.add_action_catch(float(actions[local_idx][0]), float(actions[local_idx][1]))
                print(
                    "[DP grasp] episode=%d step=%d reward=%.3f action=(%.4f, %.4f) reason=%s"
                    % (
                        int(episode_idx),
                        int(steps),
                        reward,
                        float(actions[local_idx][0]),
                        float(actions[local_idx][1]),
                        str(info.get("reason", "")),
                    ),
                    flush=True,
                )

                success = bool(info.get("success", False) or info.get("goal", 0) == 1)
                invalid_abort = str(info.get("reason", "")) == "invalid_abort"
                done_flag = terminated or truncated or success or invalid_abort or steps >= cfg.max_stage_steps
                if done_flag:
                    break

            weight = _sample_weight(reward_sum, success, invalid_abort)
            if training and executed_len > 0 and weight > 0.0:
                self.grasp_agent.store_transition(DPTransition(
                    obs_image=obs_image,
                    obs_state=obs_state,
                    obs_graph=obs_graph,
                    raw_actions=raw_actions,
                    executed_len=executed_len,
                    reward_sum=float(reward_sum),
                    done=bool(done_flag),
                    success=bool(success),
                    sample_weight=float(weight),
                    info={"episode": int(episode_idx), "stage": "grasp", **last_info},
                ))

            decisions += 1
            if done_flag:
                break

        summary = StageEpisodeSummary(
            stage="grasp",
            success=bool(success),
            invalid_abort=bool(invalid_abort),
            steps=int(steps),
            decisions=int(decisions),
            episode_return=float(episode_return),
            info=last_info,
        )
        self.log_writer_catch.add(
            episode_num=episode_idx,
            return_all=float(episode_return),
            goal=1 if success else 0,
            grasp_steps=int(steps),
            grasp_decisions=int(decisions),
            grasp_info=last_info,
        )
        return summary, obs

    def run_tai_stage(
        self,
        episode_idx: int,
        start_obs: Dict[str, Any],
        *,
        training: bool,
        deterministic: bool,
    ) -> Tuple[StageEpisodeSummary, Dict[str, Any]]:
        cfg = self.cfg.tai
        obs = start_obs
        steps = 0
        decisions = 0
        episode_return = 0.0
        success = False
        last_info: Dict[str, Any] = {}

        while steps < cfg.max_stage_steps and not success:
            obs_image, obs_state, obs_graph = self._clone_obs(obs)
            chunk = self.tai_agent.act(obs_image, obs_state, obs_graph, deterministic=deterministic)
            actions = np.asarray(chunk["actions"], dtype=np.float32)
            raw_actions = np.asarray(chunk["raw_actions"], dtype=np.float32)

            reward_sum = 0.0
            executed_len = 0
            done_flag = False

            for local_idx in range(cfg.chunk_len):
                reply = self.client.step_tai(actions[local_idx])
                obs = reply["obs"]
                reward = float(reply["reward"])
                terminated = bool(reply["terminated"])
                truncated = bool(reply["truncated"])
                info = dict(reply["info"])
                last_info = info

                reward_sum += reward
                episode_return += reward
                executed_len += 1
                steps += 1

                self.log_writer_tai.add_action_tai(
                    float(actions[local_idx][0]),
                    float(actions[local_idx][1]),
                    float(actions[local_idx][2]),
                )
                print(
                    "[DP tai] episode=%d step=%d reward=%.3f action=(%.4f, %.4f, %.4f) reason=%s"
                    % (
                        int(episode_idx),
                        int(steps),
                        reward,
                        float(actions[local_idx][0]),
                        float(actions[local_idx][1]),
                        float(actions[local_idx][2]),
                        str(info.get("reason", "")),
                    ),
                    flush=True,
                )

                success = bool(info.get("success", False) or info.get("goal", 0) == 1)
                done_flag = terminated or truncated or success or steps >= cfg.max_stage_steps
                if done_flag:
                    break

            weight = _sample_weight(reward_sum, success, False)
            if training and executed_len > 0 and weight > 0.0:
                self.tai_agent.store_transition(DPTransition(
                    obs_image=obs_image,
                    obs_state=obs_state,
                    obs_graph=obs_graph,
                    raw_actions=raw_actions,
                    executed_len=executed_len,
                    reward_sum=float(reward_sum),
                    done=bool(done_flag),
                    success=bool(success),
                    sample_weight=float(weight),
                    info={"episode": int(episode_idx), "stage": "tai", **last_info},
                ))

            decisions += 1
            if done_flag:
                break

        summary = StageEpisodeSummary(
            stage="tai",
            success=bool(success),
            invalid_abort=False,
            steps=int(steps),
            decisions=int(decisions),
            episode_return=float(episode_return),
            info=last_info,
        )
        self.log_writer_tai.add(
            episode_num=episode_idx,
            return_all=float(episode_return),
            goal=1 if success else 0,
            tai_steps=int(steps),
            tai_decisions=int(decisions),
            tai_info=last_info,
        )
        return summary, obs

    def _record_episode(self, summary: Dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_make_json_safe(summary), ensure_ascii=False) + "\n")

    def _write_summary(self, summary: Dict[str, Any]) -> None:
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(_make_json_safe(summary), f, ensure_ascii=False, indent=2)

    def _save_stage_by_test_score(
        self,
        agent: OnlineDiffusionAgent,
        stage_name: str,
        episode: int,
        score: float,
        extra: Dict[str, Any],
    ) -> str:
        stage_dir = self.save_dir / "checkpoints" / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        score_tag = ("%.4f" % float(score)).replace(".", "_")
        save_path = stage_dir / ("%s_episode_%d_score_%s.ckpt" % (stage_name, int(episode), score_tag))
        return agent.save_checkpoint(str(save_path), episode=episode, score=score, extra=extra)

    def train(self) -> Dict[str, float]:
        start_time = time.time()
        print("Starting pure online Diffusion Policy training.", flush=True)
        print("Catch log: %s" % self.catch_log_path, flush=True)
        print("Tai log: %s" % self.tai_log_path, flush=True)
        try:
            for episode in range(1, self.cfg.total_episodes + 1):
                obs, reset_info = self.client.reset(
                    seed=self.cfg.seed + episode,
                    options=self.cfg.make_reset_options(),
                )
                grasp_summary, obs = self.run_grasp_stage(
                    episode_idx=episode,
                    start_obs=obs,
                    training=True,
                    deterministic=False,
                )
                self.grasp_episode_count += 1
                grasp_ma = self.grasp_success_tracker.add(1.0 if grasp_summary.success else 0.0)

                if grasp_summary.success:
                    tai_summary, obs = self.run_tai_stage(
                        episode_idx=episode,
                        start_obs=obs,
                        training=True,
                        deterministic=False,
                    )
                    self.tai_episode_count += 1
                    tai_ma = self.tai_success_tracker.add(1.0 if tai_summary.success else 0.0)
                else:
                    tai_summary = None
                    tai_ma = self.tai_success_tracker.mean

                pipeline_success = bool(grasp_summary.success and tai_summary is not None and tai_summary.success)
                pipeline_ma = self.pipeline_success_tracker.add(1.0 if pipeline_success else 0.0)

                grasp_update = self._maybe_update_agent(self.grasp_agent, self.cfg.grasp, self.grasp_episode_count)
                tai_update = (
                    self._maybe_update_agent(self.tai_agent, self.cfg.tai, self.tai_episode_count)
                    if self.tai_episode_count > 0
                    else {"updated": 0.0, "diffusion_loss": 0.0}
                )

                episode_summary = {
                    "episode": int(episode),
                    "reset_info": reset_info,
                    "grasp_success": bool(grasp_summary.success),
                    "tai_success": None if tai_summary is None else bool(tai_summary.success),
                    "pipeline_success": bool(pipeline_success),
                    "grasp_steps": int(grasp_summary.steps),
                    "tai_steps": 0 if tai_summary is None else int(tai_summary.steps),
                    "grasp_return": float(grasp_summary.episode_return),
                    "tai_return": 0.0 if tai_summary is None else float(tai_summary.episode_return),
                    "total_return": float(grasp_summary.episode_return + (0.0 if tai_summary is None else tai_summary.episode_return)),
                    "grasp_ma": float(grasp_ma),
                    "tai_ma": float(tai_ma),
                    "pipeline_ma": float(pipeline_ma),
                    "grasp_update": grasp_update,
                    "tai_update": tai_update,
                    "grasp_info": grasp_summary.info,
                    "tai_info": None if tai_summary is None else tai_summary.info,
                }
                self._record_episode(episode_summary)
                self.log_writer_catch.add(
                    diffusion_loss=float(grasp_update.get("diffusion_loss", 0.0)),
                    grasp_buffer_size=float(grasp_update.get("buffer_size", len(self.grasp_agent.buffer))),
                )
                self.log_writer_catch.save_catch(self.catch_log_path)
                self.log_writer_catch.clear()
                if tai_summary is not None:
                    self.log_writer_tai.add(
                        diffusion_loss_tai=float(tai_update.get("diffusion_loss", 0.0)),
                        tai_buffer_size=float(tai_update.get("buffer_size", len(self.tai_agent.buffer))),
                    )
                    self.log_writer_tai.save_tai(self.tai_log_path)
                    self.log_writer_tai.clear()

                print(
                    "[DP episode %d] total_return=%.3f grasp=%d tai=%s pipeline=%d "
                    "grasp_loss=%.6f tai_loss=%.6f"
                    % (
                        int(episode),
                        float(episode_summary["total_return"]),
                        int(grasp_summary.success),
                        "None" if tai_summary is None else str(int(tai_summary.success)),
                        int(pipeline_success),
                        float(grasp_update.get("diffusion_loss", 0.0)),
                        float(tai_update.get("diffusion_loss", 0.0)),
                    ),
                    flush=True,
                )

                if episode % self.cfg.test_interval == 0:
                    test_result = self._run_eval_loop(num_episodes=self.cfg.num_test_episodes, deterministic=True)
                    score = float(
                        test_result["pipeline_success_rate"]
                        if self.cfg.ranking_metric == "pipeline"
                        else test_result["grasp_success_rate"]
                    )
                    if episode % self.cfg.save_interval == 0 and score > self.best_test_score:
                        self.best_test_score = score
                        paths = [
                            self._save_stage_by_test_score(
                                self.grasp_agent,
                                "grasp",
                                episode,
                                score,
                                {"test_result": test_result},
                            )
                        ]
                        if self.tai_episode_count > 0:
                            paths.append(
                                self._save_stage_by_test_score(
                                    self.tai_agent,
                                    "tai",
                                    episode,
                                    score,
                                    {"test_result": test_result},
                                )
                            )
                        self.model_ranking.add_and_manage(score, paths, episode)
        finally:
            elapsed = time.time() - start_time
            final_summary = {
                "episodes": float(self.cfg.total_episodes),
                "grasp_success_rate_ma": float(self.grasp_success_tracker.mean),
                "tai_success_rate_ma": float(self.tai_success_tracker.mean),
                "pipeline_success_rate_ma": float(self.pipeline_success_tracker.mean),
                "elapsed_seconds": float(elapsed),
            }
            self._write_summary(final_summary)
            self.close()

        return {
            "episodes": float(self.cfg.total_episodes),
            "grasp_success_rate_ma": float(self.grasp_success_tracker.mean),
            "tai_success_rate_ma": float(self.tai_success_tracker.mean),
            "pipeline_success_rate_ma": float(self.pipeline_success_tracker.mean),
        }

    @torch.no_grad()
    def _run_eval_loop(self, num_episodes: int = 20, deterministic: bool = True) -> Dict[str, float]:
        grasp_successes: List[int] = []
        tai_successes: List[int] = []
        pipeline_successes: List[int] = []
        returns: List[float] = []
        for episode in range(1, int(num_episodes) + 1):
            obs, _ = self.client.reset(
                seed=self.cfg.seed + 100000 + episode,
                options=self.cfg.make_reset_options(),
            )
            grasp_summary, obs = self.run_grasp_stage(
                episode_idx=episode,
                start_obs=obs,
                training=False,
                deterministic=deterministic,
            )
            if grasp_summary.success:
                tai_summary, obs = self.run_tai_stage(
                    episode_idx=episode,
                    start_obs=obs,
                    training=False,
                    deterministic=deterministic,
                )
            else:
                tai_summary = None

            pipeline_success = bool(grasp_summary.success and tai_summary is not None and tai_summary.success)
            total_return = grasp_summary.episode_return + (0.0 if tai_summary is None else tai_summary.episode_return)
            grasp_successes.append(int(grasp_summary.success))
            tai_successes.append(0 if tai_summary is None else int(tai_summary.success))
            pipeline_successes.append(int(pipeline_success))
            returns.append(float(total_return))
        return {
            "episodes": float(num_episodes),
            "grasp_success_rate": float(np.mean(grasp_successes)) if grasp_successes else 0.0,
            "tai_success_rate": float(np.mean(tai_successes)) if tai_successes else 0.0,
            "pipeline_success_rate": float(np.mean(pipeline_successes)) if pipeline_successes else 0.0,
            "mean_total_return": float(np.mean(returns)) if returns else 0.0,
        }

    @torch.no_grad()
    def evaluate(self, num_episodes: int = 20, deterministic: bool = True) -> Dict[str, float]:
        try:
            return self._run_eval_loop(num_episodes=num_episodes, deterministic=deterministic)
        finally:
            self.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pure Diffusion Policy trainer for Webots")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default=str(resolve_device()))
    parser.add_argument("--save-dir", type=str, default="python_scripts/DiffusionPolicy/checkpoint")
    parser.add_argument("--exp-name", type=str, default="pure_diffusion_policy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grasp-ckpt", type=str, default="")
    parser.add_argument("--tai-ckpt", type=str, default="")
    parser.add_argument("--keep-server", action="store_true", help="do not send close() to Webots when trainer exits")
    parser.add_argument("--test-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--num-test-episodes", type=int, default=50)
    parser.add_argument("--ranking-top-n", type=int, default=5)
    parser.add_argument("--ranking-metric", type=str, default="grasp", choices=["grasp", "pipeline"])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = TrainerConfig(
        host=args.host,
        port=args.port,
        device=args.device,
        total_episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        exp_name=args.exp_name,
        save_dir=args.save_dir,
        close_server_on_exit=not args.keep_server,
        seed=args.seed,
        test_interval=args.test_interval,
        save_interval=args.save_interval,
        num_test_episodes=args.num_test_episodes,
        ranking_top_n=args.ranking_top_n,
        ranking_metric=args.ranking_metric,
    )
    trainer = TwoStageSocketTrainer(cfg)
    trainer.load_stage_checkpoints(
        grasp_checkpoint=args.grasp_ckpt or None,
        tai_checkpoint=args.tai_ckpt or None,
    )
    if args.mode == "train":
        result = trainer.train()
    else:
        result = trainer.evaluate(num_episodes=args.eval_episodes, deterministic=True)
    print(json.dumps(_make_json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
