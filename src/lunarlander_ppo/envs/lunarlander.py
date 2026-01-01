from __future__ import annotations

from typing import Optional

import gymnasium as gym

from lunarlander_ppo.utils.seeding import seed_everything


def make_env(env_name: str, seed: Optional[int] = None, render_mode: Optional[str] = None) -> gym.Env:
    """
    Factory for LunarLander (or other gymnasium envs) with consistent seeding.
    """
    env = gym.make(env_name, render_mode=render_mode)

    if seed is not None:
        # Seed everything we can
        seed_everything(seed)
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except Exception:
            pass

    return env
