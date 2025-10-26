"""MAML/Reptile-style meta-learning utilities.

This module provides a lightweight, robust implementation of a first-order
Reptile-style meta-update for policy networks, suitable for meta-RL settings.
It focuses on reliability, clear typing/docs, and defensive programming.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

TensorDict = Mapping[str, torch.Tensor]


class MAMLMetaLearner(nn.Module):
    """First-order Reptile-like meta-learner for policy networks.

    Args:
        policy_network: A torch.nn.Module that maps states -> action logits.
        meta_lr: Outer-loop interpolation step-size.
        inner_lr: Inner-loop adaptation learning rate.
        num_inner_steps: Number of gradient steps for adaptation per task.
        device: Device to run computations on. If None, inferred from network.

    Notes:
        - This implementation uses a Reptile-style outer update for stability
          and simplicity (no second-order gradients).
        - Expects task batches to provide tensors with keys: 'states',
          'actions', 'rewards'. Shapes should be broadcastable; actions should
          be Long dtype for Categorical log_prob.
    """

    def __init__(
        self,
        policy_network: nn.Module,
        meta_lr: float = 1e-3,
        inner_lr: float = 1e-2,
        num_inner_steps: int = 5,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if not isinstance(policy_network, nn.Module):
            raise TypeError("policy_network must be a torch.nn.Module")
        if meta_lr <= 0 or inner_lr <= 0:
            raise ValueError("meta_lr and inner_lr must be positive")
        if num_inner_steps <= 0:
            raise ValueError("num_inner_steps must be > 0")

        self.policy_network = policy_network
        self.meta_lr = float(meta_lr)
        self.inner_lr = float(inner_lr)
        self.num_inner_steps = int(num_inner_steps)
        self.device = device or next(policy_network.parameters()).device
        self.to(self.device)

    @torch.no_grad()
    def _snapshot_params(self) -> Dict[str, torch.Tensor]:
        return {n: p.detach().clone() for n, p in self.policy_network.named_parameters()}

    def _validate_task_batch(self, task_batch: Iterable[TensorDict]) -> List[TensorDict]:
        if task_batch is None:
            raise ValueError("task_batch cannot be None")
        batch_list: List[TensorDict] = list(task_batch)
        if len(batch_list) == 0:
            raise ValueError("task_batch is empty")
        for i, td in enumerate(batch_list):
            for key in ("states", "actions", "rewards"):
                if key not in td:
                    raise KeyError(f"task_batch[{i}] missing key: {key}")
            if td["actions"].dtype != torch.long:
                raise TypeError("actions must be torch.long for Categorical")
        return batch_list

    def meta_update(self, task_batch: Iterable[TensorDict]) -> float:
        """Run inner-loop adaptation on each task and apply Reptile meta-update.

        Returns the average inner-loop loss across all steps and tasks.
        """
        batch_list = self._validate_task_batch(task_batch)
        initial_params = self._snapshot_params()
        total_loss = 0.0

        for td in batch_list:
            # Move tensors to device defensively
            states = td["states"].to(self.device)
            actions = td["actions"].to(self.device)
            rewards = td["rewards"].to(self.device)

            if states.ndim == 0 or actions.ndim == 0 or rewards.ndim == 0:
                raise ValueError("states/actions/rewards must be tensors with at least 1 dimension")
            if states.shape[0] != actions.shape[0] or actions.shape[0] != rewards.shape[0]:
                raise ValueError("Batch size mismatch among states/actions/rewards")

            # Fresh optimizer for task adaptation
            task_optimizer = optim.SGD(self.policy_network.parameters(), lr=self.inner_lr)

            for _ in range(self.num_inner_steps):
                task_optimizer.zero_grad(set_to_none=True)
                try:
                    logits = self.policy_network(states)
                    if not torch.is_tensor(logits):
                        raise TypeError("policy_network must return a tensor of logits")
                    dist = Categorical(logits=logits)
                    log_probs = dist.log_prob(actions)
                    loss = -(log_probs * rewards).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite loss encountered")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
                    task_optimizer.step()
                    total_loss += float(loss.detach().item())
                except Exception as e:
                    # Fail fast with context; do not leave partial grads around
                    for p in self.policy_network.parameters():
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()
                    raise RuntimeError(f"Inner-loop step failed: {type(e).__name__}: {e}") from e

            # Reptile meta-update: interpolate toward adapted params
            with torch.no_grad():
                for name, param in self.policy_network.named_parameters():
                    base = initial_params[name]
                    if param.data.shape != base.shape:
                        raise RuntimeError(f"Param shape changed during adaptation: {name}")
                    param.data = base + self.meta_lr * (param.data - base)

        denom = len(batch_list) * self.num_inner_steps
        return float(total_loss / max(denom, 1))


__all__ = ["MAMLMetaLearner"]
