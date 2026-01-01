from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
import gymnasium as gym

from lunarlander_ppo.agent.ppo import PPOAgent, PPOConfig
from lunarlander_ppo.envs.lunarlander import make_env
from lunarlander_ppo.utils.io import load_config


def main():
    parser = argparse.ArgumentParser(description="Record a rollout video from a checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="assets/videos")
    parser.add_argument("--max_timesteps", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_config(args.config)

    env_probe = make_env(cfg["env_name"], seed=cfg.get("seed", 42), render_mode=None)
    state_dim = env_probe.observation_space.shape[0]
    action_dim = env_probe.action_space.n
    env_probe.close()

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    ppo_cfg = PPOConfig(
        learning_rate=float(cfg["learning_rate"]),
        gamma=float(cfg["gamma"]),
        gae_lambda=float(cfg.get("gae_lambda", 0.95)),
        clip_epsilon=float(cfg["clip_epsilon"]),
        update_epochs=int(cfg["update_epochs"]),
        mini_batch_size=int(cfg["mini_batch_size"]),
        entropy_coef=float(cfg["entropy_coef"]),
        value_loss_coef=float(cfg["value_loss_coef"]),
        max_grad_norm=float(cfg.get("max_grad_norm", 0.5)),
        hidden_dim=int(cfg["hidden_dim"]),
    )

    agent = PPOAgent(state_dim, action_dim, ppo_cfg, device=device_t)
    agent.load(args.checkpoint)

    os.makedirs(args.out_dir, exist_ok=True)

    env = make_env(cfg["env_name"], seed=cfg.get("seed", 42), render_mode="rgb_array")

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=args.out_dir,
        episode_trigger=lambda ep: True,
        name_prefix="ppo_lunarlander",
        disable_logger=True,
    )

    state, _ = env.reset()
    done = False
    steps = 0
    total = 0.0

    while not done and steps < args.max_timesteps:
        action, _, _ = agent.act(state, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total += float(reward)
        steps += 1

    env.close()
    print(f"Recorded video. Reward={total:.2f}, steps={steps}. Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
