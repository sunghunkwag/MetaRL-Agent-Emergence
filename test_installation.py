#!/usr/bin/env python3
"""Test installation and run a quick validation experiment."""

import os
import sys
import torch
import numpy as np

def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*60)
    print("Testing imports...")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import gymnasium
        print(f"✓ Gymnasium {gymnasium.__version__}")
    except ImportError as e:
        print(f"✗ Gymnasium: {e}")
        return False
    
    try:
        import metaworld
        print(f"✓ Meta-World installed")
    except ImportError as e:
        print(f"✗ Meta-World: {e}")
        print("  Install with: pip install git+https://github.com/Farama-Foundation/Metaworld.git@master")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
        return False
    
    print("\n✓ All required packages are installed!")
    return True

def test_components():
    """Test that framework components can be imported."""
    print("\n" + "="*60)
    print("Testing framework components...")
    print("="*60)
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from meta_learners.maml_ppo import MAMLPPO, MAMLPPOPolicy
        print("✓ MAML+PPO meta-learner")
    except Exception as e:
        print(f"✗ MAML+PPO: {e}")
        return False
    
    try:
        from multi_agent.metaworld_wrapper import MetaWorldMultiAgentWrapper, MetaWorldTaskSampler
        print("✓ MetaWorld wrapper")
    except Exception as e:
        print(f"✗ MetaWorld wrapper: {e}")
        return False
    
    try:
        from ssm_policies.ssm_policy_network import SSMPolicyNetwork, SSMBlock
        print("✓ SSM policy network")
    except Exception as e:
        print(f"✗ SSM policy network: {e}")
        return False
    
    print("\n✓ All framework components are working!")
    return True

def test_metaworld_environments():
    """Test MetaWorld environment creation."""
    print("\n" + "="*60)
    print("Testing MetaWorld environments...")
    print("="*60)
    
    try:
        import metaworld
        
        # Test MT10
        print("\nTesting MT10 benchmark...")
        mt10 = metaworld.MT10(seed=42)
        train_tasks = list(mt10.train_classes.keys())
        print(f"✓ MT10: {len(train_tasks)} training tasks available")
        print(f"  Tasks: {', '.join(train_tasks[:3])}...")
        
        # Test creating an environment
        task_name = train_tasks[0]
        env_cls = mt10.train_classes[task_name]
        env = env_cls()
        print(f"✓ Created environment for '{task_name}'")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.shape}")
        env.close()
        
        return True
    except Exception as e:
        print(f"✗ MetaWorld environments: {e}")
        return False

def run_quick_test():
    """Run a very quick test to verify the framework works."""
    print("\n" + "="*60)
    print("Running quick framework test...")
    print("="*60)
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from meta_learners.maml_ppo import MAMLPPO
        from multi_agent.metaworld_wrapper import MetaWorldTaskSampler
        import torch
        
        # Initialize task sampler
        print("\nInitializing Meta-World MT10...")
        task_sampler = MetaWorldTaskSampler(benchmark_name='MT10', seed=42)
        print(f"✓ Task sampler ready with {len(task_sampler.train_tasks)} train tasks")
        
        # Get environment dimensions
        sample_task = task_sampler.train_tasks[0]
        env = task_sampler.get_task_env(sample_task)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        print(f"✓ Environment: state_dim={state_dim}, action_dim={action_dim}")
        env.close()
        
        # Initialize agent
        print("\nInitializing MAML+PPO agent...")
        agent = MAMLPPO(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,  # Small for testing
            meta_lr=1e-3,
            inner_lr=1e-2
        )
        print(f"✓ Agent initialized with {sum(p.numel() for p in agent.policy.parameters())} parameters")
        
        # Test trajectory collection
        print("\nTesting trajectory collection...")
        env = task_sampler.get_task_env(sample_task)
        state, _ = env.reset()
        
        # Collect a few steps
        episode_reward = 0
        for step in range(10):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = agent.policy.get_action(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            if terminated or truncated:
                break
        
        env.close()
        print(f"✓ Collected 10-step trajectory (reward: {episode_reward:.2f})")
        
        print("\n" + "="*60)
        print("✅ All tests passed! Framework is working correctly.")
        print("="*60)
        print("\nTo run a full experiment, use:")
        print("  python experiments/run_experiment.py --benchmark MT10 --num-epochs 10")
        print("\nFor a minimal test:")
        print("  python experiments/run_experiment.py --benchmark MT10 --num-epochs 5 --meta-batch-size 2")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("\n" + "#"*60)
    print("# MetaRL-Agent-Emergence Installation Test")
    print("#"*60)
    
    # Test imports
    if not test_imports():
        print("\n✗ Installation incomplete. Please install required packages.")
        print("  Run: pip install -r requirements.txt")
        return False
    
    # Test components
    if not test_components():
        print("\n✗ Framework components not working. Check file structure.")
        return False
    
    # Test MetaWorld
    if not test_metaworld_environments():
        print("\n✗ MetaWorld not working. Check installation.")
        return False
    
    # Run quick test
    if not run_quick_test():
        print("\n✗ Framework test failed.")
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
