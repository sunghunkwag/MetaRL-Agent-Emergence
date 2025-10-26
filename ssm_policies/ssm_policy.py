"""SSM Policy Implementations (S4/Mamba-style) for RL policies.

This module provides a skeleton PyTorch implementation of a State Space Model (SSM)
policy suitable for sequential decision-making in RL. It is structured to allow
plugging different SSM backbones (e.g., S4, Mamba) for the recurrent core.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSSMCore(nn.Module):
    """A minimal SSM-like core placeholder.
    
    This is NOT a full S4/Mamba implementation. It mimics an SSM recurrent
    mapping with gated residual updates so you can wire in a real SSM later.
    Replace this with an actual S4/Mamba block when integrating real kernels.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.state_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        # x_t, h_t: [B, H]
        u = torch.tanh(self.in_proj(x_t))  # input drive
        s = torch.tanh(self.state_proj(h_t))  # state evolution
        g = torch.sigmoid(self.gate(h_t))  # gating
        h_next = self.norm(h_t + g * (u[..., : self.hidden_dim] + s))
        return h_next


class SSMPolicy(nn.Module):
    """State Space Model-based policy for discrete or continuous actions.

    Args:
        state_dim: Input observation dimension
        action_dim: Number of discrete actions (if discrete) or dim of action head
        hidden_dim: Hidden dimension for SSM core
        continuous: If True, outputs mean/log_std for Gaussian policy
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = False,
    ):
        super().__init__()
        self.continuous = continuous
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.core = SimpleSSMCore(hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        return torch.zeros(batch_size, self.policy_head.in_features, device=device)

    def forward(self, obs: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        # obs: [B, T, state_dim] or [B, state_dim]
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        B, T, _ = obs.shape
        if hidden is None:
            hidden = self.init_hidden(B, obs.device)

        logits_seq = []
        h = hidden
        for t in range(T):
            x_t = self.state_encoder(obs[:, t])
            h = self.core(x_t, h)
            logits = self.policy_head(h)
            logits_seq.append(logits)

        logits = torch.stack(logits_seq, dim=1)  # [B, T, A]
        if T == 1:
            logits = logits.squeeze(1)

        if self.continuous:
            return logits, self.log_std.expand_as(logits), h
        else:
            return logits, h


if __name__ == "__main__":
    # Quick smoke test
    B, T, S = 4, 5, 16
    A = 6
    model = SSMPolicy(state_dim=S, action_dim=A, hidden_dim=64, continuous=False)
    obs = torch.randn(B, T, S)
    logits, h = model(obs)
    print(logits.shape, h.shape)  # Expect [B, T, A], [B, H]
