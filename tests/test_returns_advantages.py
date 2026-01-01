import numpy as np
import torch

from lunarlander_ppo.agent.ppo import PPOAgent, PPOConfig, RolloutBuffer


def test_update_runs_smoke():
    cfg = PPOConfig(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        update_epochs=2,
        mini_batch_size=4,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        hidden_dim=32,
    )

    device = torch.device("cpu")
    agent = PPOAgent(state_dim=8, action_dim=4, cfg=cfg, device=device)

    buf = RolloutBuffer()
    state = np.zeros(8, dtype=np.float32)

    # Fake tiny rollout
    for _ in range(8):
        action, logp, value = agent.act(state, deterministic=False)
        reward = 1.0
        done = 0.0
        buf.add(state, action, reward, done, logp, value)

    metrics = agent.update(buf, next_state=state)
    assert "loss" in metrics
    assert isinstance(metrics["loss"], float)
