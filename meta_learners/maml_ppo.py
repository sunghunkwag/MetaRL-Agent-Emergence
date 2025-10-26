import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from torch.distributions import Categorical

class MAMLPPOPolicy(nn.Module):
    """MAML-adapted PPO policy network with support for fast adaptation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action probabilities and value."""
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value
    
    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

class MAMLPPO:
    """Model-Agnostic Meta-Learning with PPO for multi-agent settings."""
    
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
        max_grad_norm: float = 0.5
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize meta-policy
        self.policy = MAMLPPOPolicy(state_dim, action_dim, hidden_dim)
        self.meta_optimizer = optim.Adam(self.policy.parameters(), lr=meta_lr)
        
    def inner_loop_adapt(
        self,
        trajectories: List[Dict],
        num_inner_steps: int = 5
    ) -> MAMLPPOPolicy:
        """Fast adaptation using inner loop gradient updates."""
        # Clone policy for adaptation
        adapted_policy = MAMLPPOPolicy(
            self.state_dim, self.action_dim
        )
        adapted_policy.load_state_dict(self.policy.state_dict())
        
        inner_optimizer = optim.SGD(adapted_policy.parameters(), lr=self.inner_lr)
        
        for _ in range(num_inner_steps):
            loss = self._compute_ppo_loss(adapted_policy, trajectories)
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
        
        return adapted_policy
    
    def _compute_ppo_loss(
        self,
        policy: MAMLPPOPolicy,
        trajectories: List[Dict]
    ) -> torch.Tensor:
        """Compute PPO loss with clipping."""
        states = torch.stack([t['state'] for t in trajectories])
        actions = torch.tensor([t['action'] for t in trajectories])
        old_log_probs = torch.stack([t['log_prob'] for t in trajectories])
        returns = torch.tensor([t['return'] for t in trajectories], dtype=torch.float32)
        advantages = torch.tensor([t['advantage'] for t in trajectories], dtype=torch.float32)
        
        # Get current policy predictions
        action_probs, values = policy(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Compute ratio for PPO clipping
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(), returns)
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        return loss
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        dones: List[bool],
        next_value: torch.Tensor
    ) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        gae = 0
        
        values = [v.item() for v in values] + [next_value.item()]
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = values[t + 1] if not dones[t] else 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - dones[t])
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return returns, advantages
    
    def meta_update(
        self,
        task_trajectories: List[List[Dict]]
    ):
        """Perform meta-update across multiple tasks."""
        meta_loss = 0
        
        for trajectories in task_trajectories:
            # Adapt policy to task
            adapted_policy = self.inner_loop_adapt(trajectories)
            
            # Compute loss on adapted policy
            loss = self._compute_ppo_loss(adapted_policy, trajectories)
            meta_loss += loss
        
        # Meta-gradient update
        meta_loss = meta_loss / len(task_trajectories)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
