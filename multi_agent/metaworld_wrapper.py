"""Meta-World multi-agent wrappers with defensive checks and Gymnasium API.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import gymnasium as gym
import metaworld


class MetaWorldMultiAgentWrapper:
    """Wrapper for Meta-World environments to support multi-agent meta-RL training.

    Provides utilities to sample tasks, reset per-task envs for N agents,
    and step them in lockstep, returning Gymnasium-style tuples.
    """

    def __init__(
        self,
        benchmark_name: str = "MT10",
        num_agents: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        self.benchmark_name = benchmark_name
        self.num_agents = int(num_agents)
        self.seed = seed

        # Initialize Meta-World benchmark
        if benchmark_name == "MT10":
            self.benchmark = metaworld.MT10(seed=seed)
        elif benchmark_name == "MT50":
            self.benchmark = metaworld.MT50(seed=seed)
        elif benchmark_name == "ML1":
            self.benchmark = metaworld.ML1("pick-place-v2", seed=seed)
        elif benchmark_name == "ML10":
            self.benchmark = metaworld.ML10(seed=seed)
        elif benchmark_name == "ML45":
            self.benchmark = metaworld.ML45(seed=seed)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        self.task_names = list(self.benchmark.train_classes.keys())
        if len(self.task_names) == 0:
            raise RuntimeError("No tasks available in benchmark")
        self.current_task_idx = 0
        self.envs: List[gym.Env] | None = None

    def sample_tasks(self, num_tasks: int) -> List[str]:
        """Sample random tasks from the benchmark."""
        import random
        if num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        return random.sample(self.task_names, min(num_tasks, len(self.task_names)))

    def create_env(self, task_name: str, agent_id: int = 0) -> gym.Env:
        """Create a single environment for a specific task."""
        if task_name not in self.benchmark.train_classes:
            raise KeyError(f"Unknown task name: {task_name}")
        env_cls = self.benchmark.train_classes[task_name]
        env = env_cls()
        # Sample task configuration
        tasks = [t for t in self.benchmark.train_tasks if t.env_name == task_name]
        if tasks:
            task = tasks[agent_id % len(tasks)]
            env.set_task(task)
        return env

    def reset_task(self, task_name: str) -> List[np.ndarray]:
        """Reset environments for all agents with a specific task."""
        self.envs = [self.create_env(task_name, i) for i in range(self.num_agents)]
        states: List[np.ndarray] = []
        for env in self.envs:
            obs, _ = env.reset(seed=self.seed)
            states.append(obs)
        return states

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], List[Dict]]:
        """Step all agent environments with discrete actions.

        Returns: (states, rewards, dones, infos)
        """
        if self.envs is None:
            raise RuntimeError("Call reset_task() before step()")
        if len(actions) != len(self.envs):
            raise ValueError("actions length must match number of envs")

        states: List[np.ndarray] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict] = []

        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            states.append(obs)
            rewards.append(float(reward))
            dones.append(done)
            infos.append(info)
        return states, rewards, dones, infos

    def close(self) -> None:
        """Close all environments."""
        if self.envs:
            for env in self.envs:
                env.close()
            self.envs = None

    def get_state_action_dims(self, task_name: Optional[str] = None) -> Tuple[int, int]:
        """Get state and action dimensions.

        Note: For continuous control in Meta-World, action_dim is typically > 1.
        This wrapper reports action space shape[0].
        """
        if task_name is None:
            task_name = self.task_names[0]
        env = self.create_env(task_name)
        try:
            state_dim = int(env.observation_space.shape[0])
            if hasattr(env.action_space, "shape") and env.action_space.shape is not None:
                action_dim = int(env.action_space.shape[0])
            else:
                action_dim = int(env.action_space.n)
        finally:
            env.close()
        return state_dim, action_dim


class MetaWorldTaskSampler:
    """Sampler for Meta-World tasks to support meta-learning."""

    def __init__(
        self,
        benchmark_name: str = "MT10",
        num_train_tasks: Optional[int] = None,
        num_test_tasks: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.benchmark_name = benchmark_name
        self.seed = seed

        # Initialize benchmark
        if benchmark_name == "MT10":
            self.benchmark = metaworld.MT10(seed=seed)
        elif benchmark_name == "MT50":
            self.benchmark = metaworld.MT50(seed=seed)
        elif benchmark_name == "ML1":
            self.benchmark = metaworld.ML1("pick-place-v2", seed=seed)
        elif benchmark_name == "ML10":
            self.benchmark = metaworld.ML10(seed=seed)
        elif benchmark_name == "ML45":
            self.benchmark = metaworld.ML45(seed=seed)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        # Split tasks into train and test
        all_tasks = list(self.benchmark.train_classes.keys())
        if len(all_tasks) == 0:
            raise RuntimeError("No tasks available in benchmark")

        if num_train_tasks is None:
            num_train_tasks = max(1, int(0.8 * len(all_tasks)))
        if num_test_tasks is None:
            num_test_tasks = max(1, len(all_tasks) - num_train_tasks)

        import random
        if seed is not None:
            random.seed(seed)
        random.shuffle(all_tasks)
        self.train_tasks = all_tasks[:num_train_tasks]
        self.test_tasks = all_tasks[num_train_tasks:num_train_tasks + num_test_tasks]

    def sample_train_tasks(self, batch_size: int) -> List[str]:
        """Sample training tasks."""
        import random
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return [random.choice(self.train_tasks) for _ in range(batch_size)]

    def sample_test_tasks(self, batch_size: int) -> List[str]:
        """Sample test tasks."""
        import random
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return [random.choice(self.test_tasks) for _ in range(batch_size)]

    def get_task_env(self, task_name: str) -> gym.Env:
        """Create environment for a specific task."""
        if task_name not in self.benchmark.train_classes:
            raise KeyError(f"Unknown task name: {task_name}")
        env_cls = self.benchmark.train_classes[task_name]
        env = env_cls()
        tasks = [t for t in self.benchmark.train_tasks if t.env_name == task_name]
        if tasks:
            env.set_task(tasks[0])
        return env


__all__ = [
    "MetaWorldMultiAgentWrapper",
    "MetaWorldTaskSampler",
]
