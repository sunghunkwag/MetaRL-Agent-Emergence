# MetaRL-Agent-Emergence

[Open in Google Colab ▶](https://colab.research.google.com/github/sunghunkwag/MetaRL-Agent-Emergence/blob/main/MetaRL_Colab_Demo.ipynb)

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
├── requirements.txt        # Dependencies
├── test_installation.py    # Installation verification script
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

# Install dependencies
pip install -U pip wheel setuptools
pip install -r requirements.txt
pip install mujoco==3.1.6 metaworld==2.0.0
```

## Colab Notes
- The Colab notebook also installs MuJoCo and MetaWorld and runs a tiny MT10 demo.
- See caveats in the notebook about MuJoCo headless, GPU runtime, and time limits.

## Citation
If you use this repository, please cite appropriately.
