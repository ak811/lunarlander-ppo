from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch

from lunarlander_ppo.agent.ppo import PPOAgent, PPOConfig
from lunarlander_ppo.envs.lunarlander import make_env
from lunarlander_ppo.training.evaluate import evaluate
from lunarlander_ppo.utils.io import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO policy.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    env = make_env(cfg["env_name"], seed=cfg.get("seed", 42), render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

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

    mean_r, std_r, metrics = evaluate(
        agent,
        env_name=cfg["env_name"],
        episodes=args.episodes,
        seed=cfg.get("seed", 42) + 123,
        deterministic=args.deterministic,
        max_timesteps=int(cfg["max_timesteps"]),
    )

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
