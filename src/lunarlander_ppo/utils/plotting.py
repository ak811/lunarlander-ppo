from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _nan_safelist(x: List[float]) -> np.ndarray:
    arr = np.array(x, dtype=np.float32)
    return arr


def plot_training_curves(history: Dict[str, List[float]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    ep = history["episode"]
    rewards = history["reward"]
    losses = history["loss"]
    actor_losses = history["actor_loss"]
    critic_losses = history["critic_loss"]
    entropies = history["entropy"]
    steps = history["steps"]
    elapsed = history["elapsed_sec"]
    eval_mean = _nan_safelist(history["eval_mean_reward"])

    # 1) Reward
    plt.figure(figsize=(10, 5))
    plt.plot(ep, rewards)
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_curve.png"))
    plt.close()

    # 2) Loss
    plt.figure(figsize=(10, 5))
    plt.plot(ep, losses, label="total")
    plt.plot(ep, actor_losses, label="actor")
    plt.plot(ep, critic_losses, label="critic")
    plt.title("Loss Curves")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"))
    plt.close()

    # 3) Entropy
    plt.figure(figsize=(10, 5))
    plt.plot(ep, entropies)
    plt.title("Policy Entropy")
    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "entropy.png"))
    plt.close()

    # 4) Steps
    plt.figure(figsize=(10, 5))
    plt.plot(ep, steps)
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "steps.png"))
    plt.close()

    # 5) Elapsed time
    plt.figure(figsize=(10, 5))
    plt.plot(ep, elapsed)
    plt.title("Elapsed Training Time")
    plt.xlabel("Episode")
    plt.ylabel("Seconds")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "elapsed_time.png"))
    plt.close()

    # 6) Eval mean reward (NaNs allowed)
    plt.figure(figsize=(10, 5))
    plt.plot(ep, eval_mean)
    plt.title("Evaluation Mean Reward (periodic)")
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eval_mean_reward.png"))
    plt.close()
