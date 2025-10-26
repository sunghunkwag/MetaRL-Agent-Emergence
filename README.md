# MetaRL-Agent-Emergence

**Multi-Agent Meta-RL Framework with MAML, PPO, and SSM-based Policies**

## Overview

This repository implements a complete, runnable multi-agent meta-reinforcement learning framework featuring:

- **MAML+PPO**: Model-Agnostic Meta-Learning with Proximal Policy Optimization
- **SSM-based Policies**: State Space Model architectures for efficient sequence modeling
- **Meta-World Integration**: SOTA benchmarks (MT10, MT50) for multi-task learning
- **Complete Training Pipeline**: End-to-end experiment runner with logging and visualization

## Repository Structure

```
MetaRL-Agent-Emergence/
├── meta_learners/          # Meta-learning algorithms
│   └── maml_ppo.py        # MAML+PPO implementation
├── multi_agent/            # Multi-agent environments
│   └── metaworld_wrapper.py  # Meta-World environment wrapper
├── ssm_policies/           # State Space Model policies
│   └── ssm_policy_network.py # SSM-based policy architecture
├── experiments/            # Training scripts
│   └── run_experiment.py  # Main experiment runner
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/sunghunkwag/MetaRL-Agent-Emergence.git
cd MetaRL-Agent-Emergence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Meta-World (if not automatically installed)
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master
```

## Quick Start

Run a basic meta-RL training session on Meta-World MT10:

```bash
python experiments/run_experiment.py \
    --benchmark MT10 \
    --num-epochs 50 \
    --meta-batch-size 4 \
    --output-dir ./results
```

## Usage

### Running Experiments

The main experiment runner supports various configurations:

```bash
python experiments/run_experiment.py [OPTIONS]
```

#### Key Arguments:

**Environment:**
- `--benchmark`: Meta-World benchmark [MT10, MT50, ML1, ML10, ML45] (default: MT10)
- `--seed`: Random seed (default: 42)

**Training:**
- `--num-epochs`: Number of meta-training epochs (default: 100)
- `--meta-batch-size`: Number of tasks per meta-batch (default: 4)
- `--inner-batch-size`: Trajectories per task for inner loop (default: 10)
- `--max-steps-per-episode`: Maximum steps per episode (default: 200)

**Model:**
- `--hidden-dim`: Hidden layer dimension (default: 128)
- `--meta-lr`: Meta-learning rate (default: 1e-3)
- `--inner-lr`: Inner loop learning rate (default: 1e-2)

**PPO:**
- `--gamma`: Discount factor (default: 0.99)
- `--gae-lambda`: GAE lambda (default: 0.95)
- `--clip-epsilon`: PPO clipping epsilon (default: 0.2)

**Evaluation:**
- `--eval-interval`: Evaluate every N epochs (default: 10)
- `--num-eval-tasks`: Number of evaluation tasks (default: 5)
- `--eval-episodes`: Episodes per task for evaluation (default: 3)

**Output:**
- `--output-dir`: Output directory for results (default: ./results)

### Example: Full MT10 Experiment

```bash
python experiments/run_experiment.py \
    --benchmark MT10 \
    --num-epochs 100 \
    --meta-batch-size 8 \
    --inner-batch-size 20 \
    --hidden-dim 256 \
    --meta-lr 0.001 \
    --inner-lr 0.01 \
    --eval-interval 10 \
    --output-dir ./results/mt10_full
```

### Example: Quick Test Run

```bash
python experiments/run_experiment.py \
    --benchmark MT10 \
    --num-epochs 10 \
    --meta-batch-size 2 \
    --inner-batch-size 5 \
    --eval-interval 5 \
    --output-dir ./results/test
```

## Components

### 1. MAML+PPO Meta-Learner (`meta_learners/maml_ppo.py`)

Implements Model-Agnostic Meta-Learning with Proximal Policy Optimization:
- Fast adaptation through inner loop gradient updates
- PPO for stable policy optimization
- Generalized Advantage Estimation (GAE)
- Gradient clipping for training stability

### 2. Meta-World Wrapper (`multi_agent/metaworld_wrapper.py`)

Provides interface to Meta-World benchmarks:
- Multi-agent environment wrapper
- Task sampling for meta-learning
- Support for MT10, MT50, ML1, ML10, ML45 benchmarks
- Easy integration with meta-learning algorithms

### 3. SSM Policy Network (`ssm_policies/ssm_policy_network.py`)

State Space Model-based policy architecture:
- Efficient sequence modeling with SSM blocks
- Long-range dependency modeling
- Actor-critic architecture
- Supports both single-step and sequence inputs

### 4. Experiment Runner (`experiments/run_experiment.py`)

Complete training pipeline:
- Meta-training loop with task sampling
- Inner and outer loop optimization
- Periodic evaluation on test tasks
- Automatic logging and visualization
- Model checkpointing (best and final models)
- Training curve plotting

## Results

After running an experiment, results are saved in the output directory:

```
results/experiment_YYYYMMDD_HHMMSS/
├── best_model.pt           # Best performing model checkpoint
├── final_model.pt          # Final model after training
├── results.json            # Training metrics and configuration
└── training_curves.png     # Visualization of training progress
```

### Results Format

The `results.json` file contains:
- Experiment configuration (all arguments)
- Training returns (per epoch)
- Test returns (per evaluation)
- Meta-learning losses (per epoch)

### Visualizations

The `training_curves.png` includes:
1. Training returns over epochs
2. Test returns at evaluation intervals
3. Meta-learning loss curve

## Benchmarks

### Meta-World Benchmarks

- **MT10**: 10 manipulation tasks, multi-task training
- **MT50**: 50 manipulation tasks, large-scale multi-task learning
- **ML1**: Single task for meta-learning
- **ML10**: 10 tasks for meta-learning
- **ML45**: 45 tasks for meta-learning

### Expected Performance

On MT10 with 100 epochs:
- Training return: Increases from ~0 to 50-100+
- Test return: Demonstrates generalization to unseen tasks
- Meta-loss: Decreases and stabilizes over training

## Development

### Adding New Meta-Learning Algorithms

1. Create a new file in `meta_learners/`
2. Implement the meta-learning interface:
   - `inner_loop_adapt()`: Fast adaptation
   - `meta_update()`: Meta-gradient update
3. Update `run_experiment.py` to use the new algorithm

### Adding New Environments

1. Create a wrapper in `multi_agent/`
2. Implement the environment interface:
   - `reset_task()`: Reset for a specific task
   - `step()`: Environment step
   - `sample_tasks()`: Task sampling
3. Update `run_experiment.py` to support the new environment

## Citation

If you use this code in your research, please cite:

```bibtex
@software{metarl_agent_emergence_2025,
  author = {Kwag, Sung Hun},
  title = {MetaRL-Agent-Emergence: Multi-Agent Meta-RL Framework},
  year = {2025},
  url = {https://github.com/sunghunkwag/MetaRL-Agent-Emergence}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contact

For questions or issues, please open a GitHub issue or contact through the repository.

---

**Status**: ✅ Complete runnable framework with SOTA benchmarks
