"""MAML-PPO with defensive programming and clear typing.

This module implements a simple PPO policy and a meta-learning wrapper that
performs inner-loop adaptation per task and a meta-update. It includes shape
checks, dtype validation, gradient clipping, and docstrings for reliability.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class MAMLPPOPolicy(nn.Module):
    """Actor-Critic policy used by MAML-PPO.

    The actor outputs action probabilities (Categorical). The critic outputs
    state values.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (action_probs, value) given state.

        state: Tensor[..., state_dim]
        """
        if state.ndim == 1:
            state = state.unsqueeze(0)
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob, value)."""
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.squeeze().item()), log_prob.squeeze(), value.squeeze()


class MAMLPPO:
    """MAML-style training wrapper around a PPO policy.

    Uses inner-loop SGD on task trajectories and outer-loop Adam meta-update.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        meta_lr: float = 1e-3,
        inner_lr: float = 1e-2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: torch.device | None = None,
    ) -> None:
        if any(x <= 0 for x in (meta_lr, inner_lr, gamma, gae_lambda, value_coef, max_grad_norm)):
            raise ValueError("learning rates, gamma, gae_lambda, value_coef, max_grad_norm must be > 0")
        if not (0.0 < clip_epsilon < 1.0):
            raise ValueError("clip_epsilon should be in (0, 1)")

        self.policy = MAMLPPOPolicy(state_dim, action_dim, hidden_dim)
        self.device = device or next(self.policy.parameters()).device
        self.policy.to(self.device)

        self.meta_lr = float(meta_lr)
        self.inner_lr = float(inner_lr)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_epsilon = float(clip_epsilon)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)

        self.meta_optimizer = optim.Adam(self.policy.parameters(), lr=self.meta_lr)

    def inner_loop_adapt(self, trajectories: List[Dict], num_inner_steps: int = 5) -> MAMLPPOPolicy:
        if len(trajectories) == 0:
            raise ValueError("trajectories is empty")
        adapted = MAMLPPOPolicy(self.policy.state_dim, self.policy.action_dim)
        adapted.load_state_dict(self.policy.state_dict())
        adapted.to(self.device)
        opt = optim.SGD(adapted.parameters(), lr=self.inner_lr)
        for _ in range(num_inner_steps):
            loss = self._compute_ppo_loss(adapted, trajectories)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(adapted.parameters(), self.max_grad_norm)
            opt.step()
        return adapted

    def _stack_field(self, trajectories: List[Dict], key: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        if key not in trajectories[0]:
            raise KeyError(f"trajectory missing key: {key}")
        vals = [t[key] for t in trajectories]
        if isinstance(vals[0], torch.Tensor):
            out = torch.stack(vals)
        else:
            out = torch.tensor(vals, dtype=dtype)
        return out.to(self.device)

    def _compute_ppo_loss(self, policy: MAMLPPOPolicy, trajectories: List[Dict]) -> torch.Tensor:
        states = self._stack_field(trajectories, "state")
        actions = self._stack_field(trajectories, "action", dtype=torch.long).long()
        old_log_probs = self._stack_field(trajectories, "log_prob")
        returns = self._stack_field(trajectories, "return", dtype=torch.float32)
        advantages = self._stack_field(trajectories, "advantage", dtype=torch.float32)

        action_probs, values = policy(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.functional.mse_loss(values.squeeze(-1), returns)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite PPO loss")
        return loss

    def compute_gae(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        dones: List[bool],
        next_value: torch.Tensor,
    ) -> Tuple[List[float], List[float]]:
        advantages: List[float] = []
        returns: List[float] = []
        gae = 0.0
        vals = [float(v.item()) for v in values] + [float(next_value.item())]
        for t in reversed(range(len(rewards))):
            nxt = 0.0 if dones[t] else vals[t + 1]
            delta = float(rewards[t]) + self.gamma * nxt - vals[t]
            gae = delta + self.gamma * self.gae_lambda * gae * (1.0 - float(dones[t]))
            advantages.insert(0, gae)
            returns.insert(0, gae + vals[t])
        return returns, advantages

    def meta_update(self, task_trajectories: List[List[Dict]]) -> float:
        if len(task_trajectories) == 0:
            raise ValueError("task_trajectories is empty")
        meta_loss = 0.0
        for traj in task_trajectories:
            adapted = self.inner_loop_adapt(traj)
            loss = self._compute_ppo_loss(adapted, traj)
            meta_loss += float(loss.detach().item())
        # Simple meta step: take gradient on current policy using all tasks' data
        self.meta_optimizer.zero_grad(set_to_none=True)
        # Reuse current policy to compute a surrogate loss across tasks
        total = 0.0
        for traj in task_trajectories:
            total = total + self._compute_ppo_loss(self.policy, traj)
        total = total / len(task_trajectories)
        total.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.meta_optimizer.step()
        return meta_loss / len(task_trajectories)

    def save(self, path: str) -> None:
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.meta_optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.meta_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
