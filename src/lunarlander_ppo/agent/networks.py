from __future__ import annotations

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """
    Actor-Critic with a shared trunk then separate actor/critic heads.
    Discrete actions via softmax policy.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)  # [batch]
        return logits, value
