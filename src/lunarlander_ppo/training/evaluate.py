from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from lunarlander_ppo.envs.lunarlander import make_env


def evaluate(
    agent,
    env_name: str,
    episodes: int = 10,
    seed: Optional[int] = None,
    deterministic: bool = True,
    max_timesteps: int = 1000,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Returns mean_reward, std_reward, and a metrics dict.
    """
    env = make_env(env_name, seed=seed, render_mode=None)
    rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        total = 0.0
        steps = 0

        while not done and steps < max_timesteps:
            action, _, _ = agent.act(state, deterministic=deterministic)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += float(reward)
            steps += 1

        rewards.append(total)

    env.close()
    rewards = np.array(rewards, dtype=np.float32)
    metrics = {
        "eval_mean_reward": float(rewards.mean()),
        "eval_std_reward": float(rewards.std()),
        "eval_min_reward": float(rewards.min()),
        "eval_max_reward": float(rewards.max()),
    }
    return metrics["eval_mean_reward"], metrics["eval_std_reward"], metrics
