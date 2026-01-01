from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from tqdm import trange

from lunarlander_ppo.agent.ppo import PPOAgent, PPOConfig, RolloutBuffer
from lunarlander_ppo.envs.lunarlander import make_env
from lunarlander_ppo.logging.writers import CSVWriter, JSONLWriter
from lunarlander_ppo.training.callbacks import BestModelTracker
from lunarlander_ppo.training.evaluate import evaluate
from lunarlander_ppo.utils.io import (
    dump_config,
    get_run_dir,
    save_checkpoint,
)
from lunarlander_ppo.utils.plotting import plot_training_curves
from lunarlander_ppo.utils.seeding import seed_everything


def train(cfg: Dict[str, Any]) -> str:
    """
    Main training entry.
    Returns the created run directory path.
    """
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    run_dir = get_run_dir(
        experiments_dir=cfg.get("experiments_dir", "experiments"),
        project_name=cfg.get("project_name", "lunarlander-ppo"),
        run_name=cfg.get("run_name", ""),
    )
    os.makedirs(run_dir, exist_ok=True)

    # Save config snapshot
    dump_config(cfg, os.path.join(run_dir, "config.yaml"))

    # Setup env to infer dims
    env = make_env(cfg["env_name"], seed=seed, render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Device
    device = cfg.get("device", "auto")
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    # Build agent
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

    # Logging
    csv = CSVWriter(os.path.join(run_dir, "train_metrics.csv"))
    jsonl = JSONLWriter(os.path.join(run_dir, "train_metrics.jsonl"))

    best_tracker = BestModelTracker()

    max_episodes = int(cfg["max_episodes"])
    max_timesteps = int(cfg["max_timesteps"])
    eval_every = int(cfg.get("eval_every", 50))
    eval_episodes = int(cfg.get("eval_episodes", 10))

    history = {
        "episode": [],
        "reward": [],
        "loss": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "steps": [],
        "elapsed_sec": [],
        "eval_mean_reward": [],
        "eval_std_reward": [],
    }

    start_time = time.time()

    for ep in trange(1, max_episodes + 1, desc="Training"):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        buffer = RolloutBuffer()

        while not done and steps < max_timesteps:
            action, logp, value = agent.act(state, deterministic=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(
                state=state,
                action=action,
                reward=reward,
                done=1.0 if done else 0.0,
                log_prob=logp,
                value=value,
            )

            state = next_state
            total_reward += float(reward)
            steps += 1

        # Update
        update_metrics = agent.update(buffer, next_state=state)

        elapsed = time.time() - start_time

        # Periodic evaluation
        eval_mean = None
        eval_std = None
        if ep % eval_every == 0:
            eval_mean, eval_std, _ = evaluate(
                agent,
                env_name=cfg["env_name"],
                episodes=eval_episodes,
                seed=seed + 999,
                deterministic=True,
                max_timesteps=max_timesteps,
            )

        # Save best
        is_best = best_tracker.maybe_save(
            reward=total_reward,
            save_dir=os.path.join(run_dir, "models"),
            save_fn=lambda path: agent.save(path),
        )
        if is_best:
            # Also save a named checkpoint
            save_checkpoint(agent, os.path.join(run_dir, "models", f"best_ep_{ep}.pth"))

        # Save final checkpoint periodically (lightweight)
        if ep % max(1, eval_every) == 0:
            save_checkpoint(agent, os.path.join(run_dir, "models", "latest.pth"))

        row = {
            "episode": ep,
            "reward": total_reward,
            "steps": steps,
            "elapsed_sec": elapsed,
            **update_metrics,
            "eval_mean_reward": eval_mean if eval_mean is not None else "",
            "eval_std_reward": eval_std if eval_std is not None else "",
            "is_best": int(is_best),
        }
        csv.write(row)
        jsonl.write(row)

        # History for plots
        history["episode"].append(ep)
        history["reward"].append(total_reward)
        history["steps"].append(steps)
        history["elapsed_sec"].append(elapsed)
        history["loss"].append(update_metrics["loss"])
        history["actor_loss"].append(update_metrics["actor_loss"])
        history["critic_loss"].append(update_metrics["critic_loss"])
        history["entropy"].append(update_metrics["entropy"])
        history["eval_mean_reward"].append(eval_mean if eval_mean is not None else float("nan"))
        history["eval_std_reward"].append(eval_std if eval_std is not None else float("nan"))

        # Plot occasionally so you can actually see learning without praying
        if ep % eval_every == 0:
            plot_training_curves(history, out_dir=os.path.join(run_dir, "plots"))

    # Final save
    save_checkpoint(agent, os.path.join(run_dir, "models", "final.pth"))
    plot_training_curves(history, out_dir=os.path.join(run_dir, "plots"))

    csv.close()
    jsonl.close()
    env.close()

    return run_dir
