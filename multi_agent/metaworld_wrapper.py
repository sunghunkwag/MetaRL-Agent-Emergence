import gymnasium as gym
import metaworld
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch

class MetaWorldMultiAgentWrapper:
    """Wrapper for Meta-World environments to support multi-agent meta-RL training."""
    
    def __init__(
        self,
        benchmark_name: str = 'MT10',
        num_agents: int = 1,
        seed: Optional[int] = None
    ):
        """
        Args:
            benchmark_name: Meta-World benchmark (MT10, MT50, ML1, ML10, ML45)
            num_agents: Number of parallel agents
            seed: Random seed
        """
        self.benchmark_name = benchmark_name
        self.num_agents = num_agents
        self.seed = seed
        
        # Initialize Meta-World benchmark
        if benchmark_name == 'MT10':
            self.benchmark = metaworld.MT10(seed=seed)
        elif benchmark_name == 'MT50':
            self.benchmark = metaworld.MT50(seed=seed)
        elif benchmark_name == 'ML1':
            self.benchmark = metaworld.ML1('pick-place-v2', seed=seed)
        elif benchmark_name == 'ML10':
            self.benchmark = metaworld.ML10(seed=seed)
        elif benchmark_name == 'ML45':
            self.benchmark = metaworld.ML45(seed=seed)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        self.task_names = list(self.benchmark.train_classes.keys())
        self.current_task_idx = 0
        self.envs = None
        
    def sample_tasks(self, num_tasks: int) -> List[str]:
        """Sample random tasks from the benchmark."""
        import random
        return random.sample(self.task_names, min(num_tasks, len(self.task_names)))
    
    def create_env(self, task_name: str, agent_id: int = 0) -> gym.Env:
        """Create a single environment for a specific task."""
        env_cls = self.benchmark.train_classes[task_name]
        env = env_cls()
        
        # Sample task configuration
        tasks = [task for task in self.benchmark.train_tasks if task.env_name == task_name]
        if tasks:
            task = tasks[agent_id % len(tasks)]
            env.set_task(task)
        
        return env
    
    def reset_task(self, task_name: str) -> List[np.ndarray]:
        """Reset environments for all agents with a specific task."""
        self.envs = [self.create_env(task_name, i) for i in range(self.num_agents)]
        states = []
        for env in self.envs:
            obs, _ = env.reset(seed=self.seed)
            states.append(obs)
        return states
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], List[Dict]]:
        """Step all agent environments."""
        states, rewards, dones, infos = [], [], [], []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            states.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return states, rewards, dones, infos
    
    def close(self):
        """Close all environments."""
        if self.envs:
            for env in self.envs:
                env.close()
    
    def get_state_action_dims(self, task_name: Optional[str] = None) -> Tuple[int, int]:
        """Get state and action dimensions."""
        if task_name is None:
            task_name = self.task_names[0]
        
        env = self.create_env(task_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()
        
        return state_dim, action_dim

class MetaWorldTaskSampler:
    """Sampler for Meta-World tasks to support meta-learning."""
    
    def __init__(
        self,
        benchmark_name: str = 'MT10',
        num_train_tasks: Optional[int] = None,
        num_test_tasks: Optional[int] = None,
        seed: Optional[int] = None
    ):
        self.benchmark_name = benchmark_name
        self.seed = seed
        
        # Initialize benchmark
        if benchmark_name == 'MT10':
            self.benchmark = metaworld.MT10(seed=seed)
        elif benchmark_name == 'MT50':
            self.benchmark = metaworld.MT50(seed=seed)
        elif benchmark_name == 'ML1':
            self.benchmark = metaworld.ML1('pick-place-v2', seed=seed)
        elif benchmark_name == 'ML10':
            self.benchmark = metaworld.ML10(seed=seed)
        elif benchmark_name == 'ML45':
            self.benchmark = metaworld.ML45(seed=seed)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        # Split tasks into train and test
        all_tasks = list(self.benchmark.train_classes.keys())
        
        if num_train_tasks is None:
            num_train_tasks = int(0.8 * len(all_tasks))
        if num_test_tasks is None:
            num_test_tasks = len(all_tasks) - num_train_tasks
        
        import random
        if seed is not None:
            random.seed(seed)
        
        random.shuffle(all_tasks)
        self.train_tasks = all_tasks[:num_train_tasks]
        self.test_tasks = all_tasks[num_train_tasks:num_train_tasks + num_test_tasks]
    
    def sample_train_tasks(self, batch_size: int) -> List[str]:
        """Sample training tasks."""
        import random
        return [random.choice(self.train_tasks) for _ in range(batch_size)]
    
    def sample_test_tasks(self, batch_size: int) -> List[str]:
        """Sample test tasks."""
        import random
        return [random.choice(self.test_tasks) for _ in range(batch_size)]
    
    def get_task_env(self, task_name: str) -> gym.Env:
        """Create environment for a specific task."""
        env_cls = self.benchmark.train_classes[task_name]
        env = env_cls()
        
        # Sample task configuration
        tasks = [task for task in self.benchmark.train_tasks if task.env_name == task_name]
        if tasks:
            env.set_task(tasks[0])
        
        return env
