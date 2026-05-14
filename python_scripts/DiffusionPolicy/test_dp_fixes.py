"""Smoke tests for the pure Diffusion Policy implementation."""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))

from dp_model import DPTransition, OnlineDPConfig, OnlineDiffusionAgent


def _dummy_obs():
    image = np.random.rand(128, 128).astype("float32")
    state = np.random.randn(20).astype("float32")
    graph = state[:19].copy()
    return image, state, graph


def test_grasp_action_shape():
    cfg = OnlineDPConfig(
        name="grasp_smoke",
        action_dim=2,
        chunk_len=8,
        diffusion_steps=4,
        device="cpu",
    )
    agent = OnlineDiffusionAgent(cfg)
    image, state, graph = _dummy_obs()
    result = agent.act(image, state, graph, deterministic=False)
    actions = result["actions"]
    assert actions.shape == (8, 2), actions.shape
    assert np.max(actions) <= 1.0001
    assert np.min(actions) >= -1.0001
    print("grasp action shape ok")


def test_tai_action_shape():
    cfg = OnlineDPConfig(
        name="tai_smoke",
        action_dim=3,
        chunk_len=8,
        diffusion_steps=4,
        device="cpu",
    )
    agent = OnlineDiffusionAgent(cfg)
    image, state, graph = _dummy_obs()
    result = agent.act(image, state, graph, deterministic=True)
    actions = result["actions"]
    assert actions.shape == (8, 3), actions.shape
    assert np.max(actions) <= 1.0001
    assert np.min(actions) >= -1.0001
    print("tai action shape ok")


def test_denoising_update():
    cfg = OnlineDPConfig(
        name="update_smoke",
        action_dim=2,
        chunk_len=8,
        diffusion_steps=4,
        device="cpu",
        minibatch_size=2,
        update_epochs=1,
    )
    agent = OnlineDiffusionAgent(cfg)
    image, state, graph = _dummy_obs()
    result = agent.act(image, state, graph, deterministic=False)
    for idx in range(2):
        agent.store_transition(
            DPTransition(
                obs_image=image,
                obs_state=state,
                obs_graph=graph,
                raw_actions=result["raw_actions"],
                executed_len=8,
                reward_sum=1.0,
                done=False,
                success=False,
                sample_weight=1.0,
                info={"idx": idx},
            )
        )
    update = agent.update()
    assert update["updated"] == 1.0, update
    assert np.isfinite(update["diffusion_loss"]), update
    print("denoising update ok", update)


def main():
    test_grasp_action_shape()
    test_tai_action_shape()
    test_denoising_update()
    print("pure diffusion policy smoke tests passed")


if __name__ == "__main__":
    main()
