"""MAML (Model-Agnostic Meta-Learning) Implementation

This module implements MAML-style meta-learning for multi-agent reinforcement learning.
MAML enables fast adaptation to new tasks through gradient-based meta-learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Dict, Optional
import copy


class MAMLMetaLearner(nn.Module):
    """MAML-based meta-learner for multi-agent RL.
    
    This class implements the MAML algorithm for meta-learning across multiple
    tasks/environments. It supports first-order and second-order gradient updates.
    
    Args:
        policy_network: The policy network to be meta-learned
        meta_lr: Meta-learning rate (outer loop)
        inner_lr: Task adaptation learning rate (inner loop)
        num_inner_steps: Number of gradient steps for task adaptation
        first_order: If True, use first-order MAML (faster, less memory)
    """
    
    def __init__(
        self,
        policy_network: nn.Module,
        meta_lr: float = 1e-3,
        inner_lr: float = 1e-2,
        num_inner_steps: int = 5,
        first_order: bool = False
    ):
        super().__init__()
        self.policy_network = policy_network
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order
        
        # Meta-optimizer for outer loop updates
        self.meta_optimizer = optim.Adam(self.policy_network.parameters(), lr=meta_lr)
        
    def inner_loop_adaptation(
        self,
        task_data: Dict[str, torch.Tensor],
        adapted_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Perform inner loop adaptation on a single task.
        
        Args:
            task_data: Dictionary containing 'states', 'actions', 'rewards'
            adapted_params: Optional pre-adapted parameters to continue from
            
        Returns:
            Dictionary of adapted parameters
        """
        # Clone current parameters if not provided
        if adapted_params is None:
            adapted_params = {name: param.clone() 
                            for name, param in self.policy_network.named_parameters()}
        
        # Inner loop: adapt to task
        for step in range(self.num_inner_steps):
            # Forward pass with adapted parameters
            loss = self.compute_task_loss(task_data, adapted_params)
            
            # Compute gradients with respect to adapted parameters
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=not self.first_order,
                allow_unused=True
            )
            
            # Update adapted parameters
            adapted_params = {
                name: param - self.inner_lr * grad if grad is not None else param
                for (name, param), grad in zip(adapted_params.items(), grads)
            }
        
        return adapted_params
    
    def compute_task_loss(
        self,
        task_data: Dict[str, torch.Tensor],
        params: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute loss for a task using given parameters.
        
        Args:
            task_data: Dictionary with 'states', 'actions', 'rewards', 'dones'
            params: Optional parameter dictionary to use
            
        Returns:
            Task loss (policy gradient loss)
        """
        states = task_data['states']
        actions = task_data['actions']
        rewards = task_data['rewards']
        
        # Use functional approach if custom parameters provided
        if params is not None:
            logits = self._forward_with_params(states, params)
        else:
            logits = self.policy_network(states)
        
        # Compute policy gradient loss
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        loss = -(log_probs * rewards).mean()
        
        return loss
    
    def _forward_with_params(
        self,
        x: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass using custom parameters (functional approach)."""
        # This is a placeholder - actual implementation depends on network architecture
        # For now, temporarily set parameters and do forward pass
        original_params = {}
        for name, param in self.policy_network.named_parameters():
            original_params[name] = param.data.clone()
            param.data = params[name]
        
        output = self.policy_network(x)
        
        # Restore original parameters
        for name, param in self.policy_network.named_parameters():
            param.data = original_params[name]
        
        return output
    
    def meta_update(
        self,
        task_batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
    ) -> float:
        """Perform meta-update across a batch of tasks.
        
        Args:
            task_batch: List of (support_set, query_set) tuples for each task
            
        Returns:
            Meta-loss value
        """
        self.meta_optimizer.zero_grad()
        meta_loss = 0.0
        
        for support_set, query_set in task_batch:
            # Inner loop: adapt on support set
            adapted_params = self.inner_loop_adaptation(support_set)
            
            # Outer loop: evaluate on query set
            query_loss = self.compute_task_loss(query_set, adapted_params)
            meta_loss += query_loss
        
        # Average loss across tasks
        meta_loss = meta_loss / len(task_batch)
        
        # Meta-gradient descent
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def adapt_to_new_task(
        self,
        task_data: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None
    ) -> nn.Module:
        """Adapt the meta-learned model to a new task.
        
        Args:
            task_data: Support data for the new task
            num_steps: Number of adaptation steps (defaults to num_inner_steps)
            
        Returns:
            Adapted policy network
        """
        if num_steps is None:
            num_steps = self.num_inner_steps
        
        # Create a copy of the policy network
        adapted_policy = copy.deepcopy(self.policy_network)
        optimizer = optim.SGD(adapted_policy.parameters(), lr=self.inner_lr)
        
        # Adapt to new task
        for _ in range(num_steps):
            optimizer.zero_grad()
            loss = self.compute_task_loss(task_data, None)
            loss.backward()
            optimizer.step()
        
        return adapted_policy


class ReptileMetaLearner(nn.Module):
    """Reptile meta-learner as an alternative to MAML.
    
    Reptile is a simpler first-order meta-learning algorithm that performs
    well in practice while being more memory-efficient than MAML.
    """
    
    def __init__(
        self,
        policy_network: nn.Module,
        meta_lr: float = 1e-3,
        inner_lr: float = 1e-2,
        num_inner_steps: int = 5
    ):
        super().__init__()
        self.policy_network = policy_network
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
    
    def meta_update(
        self,
        task_batch: List[Dict[str, torch.Tensor]]
    ) -> float:
        """Reptile meta-update: move toward adapted parameters."""
        initial_params = {name: param.clone() 
                         for name, param in self.policy_network.named_parameters()}
        
        total_loss = 0.0
        
        for task_data in task_batch:
            # Adapt to task
            task_optimizer = optim.SGD(self.policy_network.parameters(), lr=self.inner_lr)
            
            for _ in range(self.num_inner_steps):
                task_optimizer.zero_grad()
                states = task_data['states']
                actions = task_data['actions']
                rewards = task_data['rewards']
                
                logits = self.policy_network(states)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                loss = -(log_probs * rewards).mean()
                
                loss.backward()
                task_optimizer.step()
                total_loss += loss.item()
            
            # Reptile update: interpolate toward adapted parameters
            with torch.no_grad():
                for name, param in self.policy_network.named_parameters():
                    param.data = initial_params[name] + self.meta_lr * (
                        param.data - initial_params[name]
                    )
        
        return total_loss / (len(task_batch) * self.num_inner_steps)
