from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lunarlander_ppo.agent.networks import ActorCritic


@dataclass
class PPOConfig:
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    update_epochs: int
    mini_batch_size: int
    entropy_coef: float
    value_loss_coef: float
    max_grad_norm: float
    hidden_dim: int


class RolloutBuffer:
    """
    Episode-length buffer (matches your original style: collect an episode then update).
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))

    def as_tensors(self, device: torch.device):
        states = torch.tensor(np.asarray(self.states), dtype=torch.float32, device=device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        values = torch.tensor(self.values, dtype=torch.float32, device=device)
        return states, actions, rewards, dones, old_log_probs, values


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: PPOConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.net = ActorCritic(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.learning_rate)

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, state]
        logits, value = self.net(state_t)
        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GAE-Lambda returns and advantages.
        values: [T], next_value: [1]
        """
        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda

        T = rewards.shape[0]
        advantages = torch.zeros(T, dtype=torch.float32, device=self.device)
        last_gae = 0.0

        # Append bootstrap
        values_ext = torch.cat([values, next_value])  # [T+1]

        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values_ext[t + 1] * nonterminal - values_ext[t]
            last_gae = delta + gamma * lam * nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return returns.detach(), advantages.detach()

    def update(self, buffer: RolloutBuffer, next_state: np.ndarray) -> Dict[str, float]:
        states, actions, rewards, dones, old_log_probs, values = buffer.as_tensors(self.device)

        with torch.no_grad():
            ns = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, next_value = self.net(ns)
            next_value = next_value.unsqueeze(0)  # [1]

        returns, advantages = self._compute_gae(rewards, dones, values, next_value)

        # Advantage normalization is boring but effective
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = states.shape[0]
        batch_size = self.cfg.mini_batch_size
        indices = torch.randperm(N, device=self.device)

        total_loss = 0.0
        total_actor = 0.0
        total_critic = 0.0
        total_entropy = 0.0
        steps = 0

        for _ in range(self.cfg.update_epochs):
            indices = indices[torch.randperm(N, device=self.device)]
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logp = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]

                logits, v_pred = self.net(mb_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.functional.mse_loss(v_pred, mb_returns)

                loss = actor_loss + self.cfg.value_loss_coef * critic_loss - self.cfg.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())
                total_actor += float(actor_loss.item())
                total_critic += float(critic_loss.item())
                total_entropy += float(entropy.item())
                steps += 1

        return {
            "loss": total_loss / max(steps, 1),
            "actor_loss": total_actor / max(steps, 1),
            "critic_loss": total_critic / max(steps, 1),
            "entropy": total_entropy / max(steps, 1),
        }

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.to(self.device)
        self.net.eval()
