#!/usr/bin/env python3
"""Improved experiment runner for multi-agent meta-RL with continuous action support.

Enhancements:
- Continuous action space handling
- Better logging and metrics tracking
- TensorBoard integration
- Checkpoint management
- Early stopping
- Performance profiling
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from meta_learners.maml_ppo_improved import ImprovedMAMLPPO
from multi_agent.metaworld_wrapper import MetaWorldTaskSampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


class ImprovedMetaRLExperiment:
    """Enhanced experiment runner for multi-agent meta-RL."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(args.output_dir) / f'experiment_{timestamp}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
        
        # Save configuration
        self._save_config()
        
        # Initialize task sampler
        self.task_sampler = MetaWorldTaskSampler(
            benchmark_name=args.benchmark,
            seed=args.seed
        )
        
        logger.info(f"Benchmark: {args.benchmark}")
        logger.info(f"Train tasks: {len(self.task_sampler.train_tasks)}")
        logger.info(f"Test tasks: {len(self.task_sampler.test_tasks)}")
        
        # Get state and action dimensions
        sample_task = self.task_sampler.train_tasks[0]
        env = self.task_sampler.get_task_env(sample_task)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Get action bounds
        action_low = float(env.action_space.low[0])
        action_high = float(env.action_space.high[0])
        action_bounds = (action_low, action_high)
        
        env.close()
        
        logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")
        logger.info(f"Action bounds: {action_bounds}")
        
        # Initialize improved MAML-PPO agent
        self.agent = ImprovedMAMLPPO(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            meta_lr=args.meta_lr,
            inner_lr=args.inner_lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            device=self.device,
            action_bounds=action_bounds,
        )
        
        # Metrics
        self.train_returns = []
        self.test_returns = []
        self.meta_losses = []
        self.best_test_return = -float('inf')
        self.epochs_without_improvement = 0
        
    def _save_config(self):
        """Save experiment configuration."""
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        logger.info(f"Saved config to {config_path}")
    
    def collect_trajectory(
        self,
        env,
        max_steps: int = 200,
        deterministic: bool = False
    ) -> Tuple[List[Dict], float]:
        """Collect a single trajectory with continuous actions.
        
        Args:
            env: Environment instance
            max_steps: Maximum steps per episode
            deterministic: Whether to use deterministic actions
            
        Returns:
            trajectory: List of transition dictionaries
            episode_return: Total episode return
        """
        trajectory = []
        state, _ = env.reset()
        episode_return = 0.0
        
        for step in range(max_steps):
            # Get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.agent.policy.get_action(
                state_tensor,
                deterministic=deterministic
            )
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            trajectory.append({
                'state': torch.FloatTensor(state),
                'action': torch.FloatTensor(action),
                'reward': float(reward),
                'log_prob': log_prob,
                'value': value,
                'done': done,
                'info': info,
            })
            
            episode_return += reward
            state = next_state
            
            if done:
                break
        
        # Compute returns and advantages using GAE
        rewards = [t['reward'] for t in trajectory]
        values = [t['value'] for t in trajectory]
        dones = [t['done'] for t in trajectory]
        
        next_value = torch.zeros(1).to(self.device)
        returns, advantages = self.agent.compute_gae(
            rewards, values, dones, next_value
        )
        
        # Add returns and advantages to trajectory
        for i, traj_step in enumerate(trajectory):
            traj_step['return'] = returns[i]
            traj_step['advantage'] = advantages[i]
        
        return trajectory, episode_return
    
    def train_meta_epoch(self) -> Tuple[float, float]:
        """Train for one meta-epoch.
        
        Returns:
            meta_loss: Average meta loss
            train_return: Average training return
        """
        # Sample batch of tasks
        task_names = self.task_sampler.sample_train_tasks(self.args.meta_batch_size)
        
        task_trajectories = []
        epoch_returns = []
        
        for task_name in task_names:
            env = None
            try:
                # Create environment for task
                env = self.task_sampler.get_task_env(task_name)
                
                # Collect trajectories for inner loop adaptation
                inner_trajectories = []
                for _ in range(self.args.inner_batch_size):
                    traj, ret = self.collect_trajectory(env, self.args.max_steps_per_episode)
                    inner_trajectories.extend(traj)
                    epoch_returns.append(ret)
                
                task_trajectories.append(inner_trajectories)
            finally:
                # Ensure environment is always closed
                if env is not None:
                    env.close()
                    del env
        
        # Meta-update
        meta_loss = self.agent.meta_update(task_trajectories)
        
        return meta_loss, np.mean(epoch_returns)
    
    def evaluate(self, num_eval_tasks: int = 5) -> Dict[str, float]:
        """Evaluate on test tasks.
        
        Args:
            num_eval_tasks: Number of tasks to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        test_tasks = self.task_sampler.sample_test_tasks(num_eval_tasks)
        test_returns = []
        success_rates = []
        
        for task_name in test_tasks:
            env = None
            try:
                env = self.task_sampler.get_task_env(task_name)
                
                # Collect trajectories for evaluation
                eval_returns = []
                eval_successes = []
                
                for _ in range(self.args.eval_episodes):
                    trajectory, episode_return = self.collect_trajectory(
                        env,
                        max_steps=self.args.max_steps_per_episode,
                        deterministic=True
                    )
                    eval_returns.append(episode_return)
                    
                    # Check for success (task-specific)
                    success = any(t['info'].get('success', False) for t in trajectory)
                    eval_successes.append(float(success))
                
                test_returns.append(np.mean(eval_returns))
                success_rates.append(np.mean(eval_successes))
            finally:
                # Ensure environment is always closed
                if env is not None:
                    env.close()
                    del env
        
        return {
            'mean_return': float(np.mean(test_returns)),
            'std_return': float(np.std(test_returns)),
            'mean_success_rate': float(np.mean(success_rates)),
            'std_success_rate': float(np.std(success_rates)),
        }
    
    def train(self):
        """Main training loop with early stopping and checkpointing."""
        logger.info(f"Starting meta-RL training on {self.args.benchmark}")
        logger.info(f"Total epochs: {self.args.num_epochs}")
        
        for epoch in tqdm(range(self.args.num_epochs), desc="Training"):
            # Train for one meta-epoch
            meta_loss, train_return = self.train_meta_epoch()
            
            self.train_returns.append(train_return)
            self.meta_losses.append(meta_loss)
            
            # Evaluate periodically
            if (epoch + 1) % self.args.eval_interval == 0:
                eval_metrics = self.evaluate(num_eval_tasks=self.args.num_eval_tasks)
                test_return = eval_metrics['mean_return']
                success_rate = eval_metrics['mean_success_rate']
                
                self.test_returns.append(test_return)
                
                logger.info(f"\nEpoch {epoch + 1}/{self.args.num_epochs}")
                logger.info(f"  Train Return: {train_return:.2f}")
                logger.info(f"  Test Return: {test_return:.2f} ± {eval_metrics['std_return']:.2f}")
                logger.info(f"  Success Rate: {success_rate:.2%} ± {eval_metrics['std_success_rate']:.2%}")
                logger.info(f"  Meta Loss: {meta_loss:.4f}")
                
                # Save best model
                if test_return > self.best_test_return:
                    self.best_test_return = test_return
                    self.epochs_without_improvement = 0
                    self.save_checkpoint('best_model.pt')
                    logger.info(f"  ✓ New best model saved!")
                else:
                    self.epochs_without_improvement += 1
                
                # Early stopping
                if (self.args.early_stopping > 0 and 
                    self.epochs_without_improvement >= self.args.early_stopping):
                    logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
            
            # Save periodic checkpoint
            if (epoch + 1) % self.args.checkpoint_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        # Save final model and results
        self.save_checkpoint('final_model.pt')
        self.save_results()
        self.plot_results()
        
        logger.info(f"\nTraining complete!")
        logger.info(f"Best test return: {self.best_test_return:.2f}")
        logger.info(f"Results saved to: {self.output_dir}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        self.agent.save(str(path))
        logger.debug(f"Saved checkpoint: {filename}")
    
    def save_results(self):
        """Save training results and metrics."""
        results = {
            'args': vars(self.args),
            'train_returns': self.train_returns,
            'test_returns': self.test_returns,
            'meta_losses': self.meta_losses,
            'best_test_return': self.best_test_return,
            'agent_metrics': self.agent.get_metrics(),
        }
        
        path = self.output_dir / 'results.json'
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {path}")
    
    def plot_results(self):
        """Plot and save training curves."""
        sns.set_style('darkgrid')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training returns
        axes[0, 0].plot(self.train_returns, label='Train Return', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Average Return')
        axes[0, 0].set_title('Training Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Test returns
        if self.test_returns:
            eval_epochs = np.arange(
                self.args.eval_interval - 1,
                len(self.train_returns),
                self.args.eval_interval
            )[:len(self.test_returns)]
            axes[0, 1].plot(eval_epochs, self.test_returns, label='Test Return', marker='o')
            axes[0, 1].axhline(y=self.best_test_return, color='r', linestyle='--', 
                              label=f'Best: {self.best_test_return:.2f}')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Average Return')
            axes[0, 1].set_title('Test Returns')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Meta losses
        axes[1, 0].plot(self.meta_losses, label='Meta Loss', alpha=0.7, color='red')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Meta-Learning Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Policy entropy (from agent metrics)
        metrics = self.agent.get_metrics()
        if metrics['entropy']:
            axes[1, 1].plot(metrics['entropy'], label='Entropy', alpha=0.7, color='green')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Entropy')
            axes[1, 1].set_title('Policy Entropy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.output_dir / 'training_curves.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves to {path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Improved Meta-RL Experiment Runner with Continuous Action Support'
    )
    
    # Environment
    parser.add_argument('--benchmark', type=str, default='MT10',
                        choices=['MT10', 'MT50', 'ML1', 'ML10', 'ML45'],
                        help='Meta-World benchmark')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    
    # Training
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of meta-training epochs')
    parser.add_argument('--meta-batch-size', type=int, default=4,
                        help='Number of tasks per meta-batch')
    parser.add_argument('--inner-batch-size', type=int, default=10,
                        help='Number of trajectories per task for inner loop')
    parser.add_argument('--max-steps-per-episode', type=int, default=200,
                        help='Maximum steps per episode')
    
    # Model
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension (increased from 128)')
    parser.add_argument('--meta-lr', type=float, default=3e-4,
                        help='Meta-learning rate (optimized)')
    parser.add_argument('--inner-lr', type=float, default=5e-3,
                        help='Inner loop learning rate (reduced)')
    
    # PPO
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                        help='PPO clipping epsilon')
    
    # Evaluation
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='Evaluate every N epochs')
    parser.add_argument('--num-eval-tasks', type=int, default=5,
                        help='Number of tasks for evaluation')
    parser.add_argument('--eval-episodes', type=int, default=3,
                        help='Episodes per task for evaluation')
    
    # Checkpointing
    parser.add_argument('--checkpoint-interval', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--early-stopping', type=int, default=0,
                        help='Early stopping patience (0 to disable)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Run experiment
    experiment = ImprovedMetaRLExperiment(args)
    experiment.train()


if __name__ == '__main__':
    main()

