#!/usr/bin/env python3
"""Run multi-agent meta-RL experiments on Meta-World benchmarks."""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from meta_learners.maml_ppo import MAMLPPO
from multi_agent.metaworld_wrapper import MetaWorldMultiAgentWrapper, MetaWorldTaskSampler

class MetaRLExperiment:
    """Experiment runner for multi-agent meta-RL."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(args.output_dir, f'experiment_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize task sampler
        self.task_sampler = MetaWorldTaskSampler(
            benchmark_name=args.benchmark,
            seed=args.seed
        )
        
        # Get state and action dimensions
        sample_task = self.task_sampler.train_tasks[0]
        env = self.task_sampler.get_task_env(sample_task)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()
        
        # Initialize MAML-PPO agent
        self.agent = MAMLPPO(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            meta_lr=args.meta_lr,
            inner_lr=args.inner_lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon
        )
        
        # Metrics
        self.train_returns = []
        self.test_returns = []
        self.meta_losses = []
        
    def collect_trajectory(
        self,
        env,
        max_steps: int = 200,
        deterministic: bool = False
    ):
        """Collect a single trajectory."""
        trajectory = []
        state, _ = env.reset()
        episode_return = 0
        
        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = self.agent.policy.get_action(state_tensor)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            trajectory.append({
                'state': state_tensor.squeeze(0),
                'action': action,
                'reward': reward,
                'log_prob': log_prob,
                'value': value,
                'done': done
            })
            
            episode_return += reward
            state = next_state
            
            if done:
                break
        
        # Compute returns and advantages using GAE
        rewards = [t['reward'] for t in trajectory]
        values = [t['value'] for t in trajectory]
        dones = [t['done'] for t in trajectory]
        
        next_value = torch.zeros(1)
        returns, advantages = self.agent.compute_gae(
            rewards, values, dones, next_value
        )
        
        for i, traj_step in enumerate(trajectory):
            traj_step['return'] = returns[i]
            traj_step['advantage'] = advantages[i]
        
        return trajectory, episode_return
    
    def train_meta_epoch(self):
        """Train for one meta-epoch."""
        # Sample batch of tasks
        task_names = self.task_sampler.sample_train_tasks(self.args.meta_batch_size)
        
        task_trajectories = []
        epoch_returns = []
        
        for task_name in task_names:
            # Create environment for task
            env = self.task_sampler.get_task_env(task_name)
            
            # Collect trajectories for inner loop adaptation
            inner_trajectories = []
            for _ in range(self.args.inner_batch_size):
                traj, ret = self.collect_trajectory(env, self.args.max_steps_per_episode)
                inner_trajectories.extend(traj)
                epoch_returns.append(ret)
            
            task_trajectories.append(inner_trajectories)
            env.close()
        
        # Meta-update
        meta_loss = self.agent.meta_update(task_trajectories)
        
        return meta_loss, np.mean(epoch_returns)
    
    def evaluate(self, num_eval_tasks: int = 5):
        """Evaluate on test tasks."""
        test_tasks = self.task_sampler.sample_test_tasks(num_eval_tasks)
        test_returns = []
        
        for task_name in test_tasks:
            env = self.task_sampler.get_task_env(task_name)
            
            # Collect trajectories for evaluation
            eval_returns = []
            for _ in range(self.args.eval_episodes):
                _, episode_return = self.collect_trajectory(
                    env,
                    max_steps=self.args.max_steps_per_episode,
                    deterministic=True
                )
                eval_returns.append(episode_return)
            
            test_returns.append(np.mean(eval_returns))
            env.close()
        
        return np.mean(test_returns)
    
    def train(self):
        """Main training loop."""
        print(f"Starting meta-RL training on {self.args.benchmark}")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        
        best_test_return = -float('inf')
        
        for epoch in tqdm(range(self.args.num_epochs), desc="Training"):
            # Train for one meta-epoch
            meta_loss, train_return = self.train_meta_epoch()
            
            self.train_returns.append(train_return)
            self.meta_losses.append(meta_loss)
            
            # Evaluate periodically
            if (epoch + 1) % self.args.eval_interval == 0:
                test_return = self.evaluate(num_eval_tasks=self.args.num_eval_tasks)
                self.test_returns.append(test_return)
                
                print(f"\nEpoch {epoch + 1}/{self.args.num_epochs}")
                print(f"  Train Return: {train_return:.2f}")
                print(f"  Test Return: {test_return:.2f}")
                print(f"  Meta Loss: {meta_loss:.4f}")
                
                # Save best model
                if test_return > best_test_return:
                    best_test_return = test_return
                    self.save_checkpoint('best_model.pt')
                    print(f"  New best model saved!")
        
        # Save final model and results
        self.save_checkpoint('final_model.pt')
        self.save_results()
        self.plot_results()
        
        print(f"\nTraining complete!")
        print(f"Best test return: {best_test_return:.2f}")
        print(f"Results saved to: {self.output_dir}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.output_dir, filename)
        self.agent.save(path)
    
    def save_results(self):
        """Save training results."""
        results = {
            'args': vars(self.args),
            'train_returns': self.train_returns,
            'test_returns': self.test_returns,
            'meta_losses': self.meta_losses,
        }
        
        path = os.path.join(self.output_dir, 'results.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def plot_results(self):
        """Plot and save training curves."""
        sns.set_style('darkgrid')
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Training returns
        axes[0].plot(self.train_returns, label='Train Return', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Average Return')
        axes[0].set_title('Training Returns')
        axes[0].legend()
        
        # Test returns
        if self.test_returns:
            eval_epochs = np.arange(0, len(self.train_returns), self.args.eval_interval)
            axes[1].plot(eval_epochs, self.test_returns, label='Test Return', marker='o')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Average Return')
            axes[1].set_title('Test Returns')
            axes[1].legend()
        
        # Meta losses
        axes[2].plot(self.meta_losses, label='Meta Loss', alpha=0.7, color='red')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Meta-Learning Loss')
        axes[2].legend()
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {path}")

def main():
    parser = argparse.ArgumentParser(description='Meta-RL Experiment Runner')
    
    # Environment
    parser.add_argument('--benchmark', type=str, default='MT10',
                        choices=['MT10', 'MT50', 'ML1', 'ML10', 'ML45'],
                        help='Meta-World benchmark')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
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
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--meta-lr', type=float, default=1e-3,
                        help='Meta-learning rate')
    parser.add_argument('--inner-lr', type=float, default=1e-2,
                        help='Inner loop learning rate')
    
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
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Run experiment
    experiment = MetaRLExperiment(args)
    experiment.train()

if __name__ == '__main__':
    main()
