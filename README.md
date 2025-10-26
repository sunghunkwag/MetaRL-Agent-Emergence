# MetaRL-Agent-Emergence
[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunghunkwag/MetaRL-Agent-Emergence/blob/main/MetaRL_Colab_Demo.ipynb)

**Multi-Agent Meta-RL Framework with MAML, PPO, and SSM-based Policies**

## Quickstart
- Click the Colab badge above to run a ready-to-go demo: clone + install + tiny MT10 run + plots.

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

# Install dependencies (includes torch, numpy, gym, gymnasium, mujoco==3.1.6,
# metaworld from git repo, stable-baselines3, wandb, and more)
pip install -U pip wheel setuptools
pip install -r requirements.txt

# Verify installation
python test_installation.py
```

### Key Dependencies
- **MuJoCo**: Version 3.1.6 (pinned for stability)
- **MetaWorld**: Installed from Farama-Foundation git repo (avoids yanked PyPI 2.0.0)
- **PyTorch**: 2.0+ for deep learning
- **Gymnasium**: 0.28+ for RL environments
- **Weights & Biases**: 0.17+ for experiment logging

## Usage

### Running Experiments
```bash
# Quick smoke test with MT10 (2 epochs)
python experiments/run_experiment.py \
    --benchmark MT10 \
    --num-epochs 2 \
    --log-dir outputs/mt10_test \
    --seed 42

# Full training run
python experiments/run_experiment.py \
    --benchmark MT10 \
    --num-epochs 100 \
    --log-dir outputs/mt10_full \
    --seed 42 \
    --use-wandb
```

### Command-Line Arguments
- `--benchmark`: Choose MT10 or MT50
- `--num-epochs`: Number of training epochs
- `--log-dir`: Output directory for logs and checkpoints
- `--seed`: Random seed for reproducibility
- `--use-wandb`: Enable Weights & Biases logging (optional)

## Colab Notes
- The Colab notebook handles all installation automatically
- Installs MuJoCo 3.1.6 and MetaWorld from git
- Runs a quick MT10 demo with 2 epochs for speed
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
