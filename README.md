# MetaRL-Agent-Emergence

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunghunkwag/MetaRL-Agent-Emergence/blob/main/MetaRL_Colab_Demo.ipynb)

**Multi-Agent Meta-RL Framework with MAML, PPO, and SSM-based Policies**

## Quickstart
- Click the Colab badge above to run a ready-to-go demo: clone + install + tiny MT10 run + plots.

## Recent Changes (2025-10-26)

### Dependencies
- **torch-rl package**: Commented out in `requirements.txt` due to compatibility issues
  - The package was causing installation conflicts and is not actively used in the current codebase

### Argument Parsing
- **experiments/run_experiment.py**: Verified argument parsing structure
  - Uses `--output-dir` argument (working correctly)
  - No `--log-dir` argument exists (code properly uses `args.output_dir` throughout)

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
├── requirements.txt        # Dependencies (includes mujoco==3.1.6, metaworld from git)
├── test_installation.py    # Installation verification script
├── MetaRL_Colab_Demo.ipynb # Google Colab demo notebook
└── README.md              # This file
```

## Installation (Local)

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

# Install dependencies (includes torch, numpy, gym, gymnasium, mujoco==3.1.6, metaworld from git)
pip install -r requirements.txt

# Verify installation
python test_installation.py
```

## Usage

### Quick Test

```bash
# Run a small-scale experiment (2 tasks, 2 epochs)
python experiments/run_experiment.py \
    --benchmark ML1 \
    --num-tasks 2 \
    --num-epochs 2 \
    --num-episodes-per-task 10 \
    --output-dir ./results
```

### Full MT10 Training

```bash
# Train on Meta-World MT10 benchmark
python experiments/run_experiment.py \
    --benchmark MT10 \
    --num-tasks 10 \
    --num-epochs 100 \
    --num-episodes-per-task 50 \
    --inner-lr 0.001 \
    --outer-lr 0.0001 \
    --output-dir ./results/mt10
```

### Command-Line Arguments

- `--benchmark`: Meta-World benchmark (ML1, ML10, MT10, MT50)
- `--num-tasks`: Number of tasks to sample
- `--num-epochs`: Meta-training epochs
- `--num-episodes-per-task`: Episodes per task during adaptation
- `--inner-lr`: Inner loop learning rate (task adaptation)
- `--outer-lr`: Outer loop learning rate (meta-update)
- `--output-dir`: Directory for saving results and checkpoints
- `--seed`: Random seed for reproducibility

## Features

### Meta-Learning (MAML+PPO)

- Fast adaptation to new tasks with few samples
- Inner loop: Task-specific policy adaptation via PPO
- Outer loop: Meta-policy optimization across tasks

### SSM-Based Policies

- Efficient sequence modeling for temporal dependencies
- Structured state space model architecture
- Better sample efficiency than standard RNNs

### Multi-Agent Coordination

- Shared meta-parameters across agents
- Independent task-specific adaptations
- Emergent collaborative behaviors

## Results

The experiment runner automatically generates:

- Training curves (meta-loss, returns)
- Per-task adaptation plots
- Checkpoint files for model weights
- JSON logs of all metrics

## Colab Demo

For a quick start without local installation:

1. Click the Colab badge at the top
2. The notebook will:
   - Clone this repository
   - Install all dependencies
   - Run a small MT10 experiment
   - Generate and display plots

**Note**: 
- See notebook for caveats about MuJoCo headless rendering, GPU runtime, and session timeouts

## Known Issues & Solutions

### MetaWorld Version
- **Issue**: PyPI metaworld 2.0.0 is yanked
- **Solution**: requirements.txt now installs from `git+https://github.com/Farama-Foundation/Metaworld.git`

### MuJoCo Compatibility
- **Issue**: Version mismatches cause rendering errors
- **Solution**: requirements.txt pins `mujoco==3.1.6` explicitly

### Colab Path Issues
- **Issue**: Files not found after git clone
- **Solution**: Notebook now includes explicit `%cd` commands to ensure correct working directory

## Troubleshooting

If you encounter import errors:

```bash
# Verify all packages installed correctly
python -c "import torch; import mujoco; import metaworld; print('All imports OK')"

# Check versions
python -c "import mujoco; print(f'MuJoCo: {mujoco.__version__}')"
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{metarl-agent-emergence,
  author = {Kwag, Sunghun},
  title = {MetaRL-Agent-Emergence: Multi-Agent Meta-RL with MAML and SSM},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/sunghunkwag/MetaRL-Agent-Emergence}
}
```

## Acknowledgments

- Meta-World benchmark from Farama Foundation
- MAML algorithm from Finn et al.
- SSM architectures inspired by recent state space model research
