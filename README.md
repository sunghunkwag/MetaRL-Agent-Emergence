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
- **MetaRL_Colab_Demo.ipynb**: Updated training demo cell
  - Changed `--log-dir` to `--output-dir` for consistency with experiment runner
  - All log path references remain consistent at `outputs/mt10_demo`

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
│   ├── __init__.py
│   └── ssm_actor_critic.py  # SSM-based policy networks
├── experiments/            # Training scripts
│   └── run_experiment.py   # Main experiment runner
├── requirements.txt        # Python dependencies
├── MetaRL_Colab_Demo.ipynb # Google Colab demo notebook
├── README.md              # This file
└── RESULTS.md             # Experimental results
```

## Features

### Meta-Learning (MAML)
- Inner loop: Fast adaptation to new tasks using gradient descent
- Outer loop: Meta-optimization across task distribution
- Supports both first-order and second-order MAML

### Policy Optimization (PPO)
- Clipped surrogate objective for stable updates
- Generalized Advantage Estimation (GAE)
- Value function learning with clipping

### State Space Models
- Efficient sequence modeling for temporal dependencies
- Lightweight alternative to Transformers
- Mamba-style selective state spaces

### Multi-Agent Support
- Meta-World MT10/MT50 benchmarks
- Task distribution sampling
- Parallel environment execution

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- MuJoCo 3.1.6 (automatically installed via pip)

### Setup

```bash
# Clone the repository
git clone https://github.com/sunghunkwag/MetaRL-Agent-Emergence.git
cd MetaRL-Agent-Emergence

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start with Google Colab
The easiest way to get started is using our Colab notebook:
1. Click the Colab badge at the top of this README
2. Run all cells sequentially
3. The notebook will:
   - Install all dependencies
   - Run a small MT10 training demo
   - Visualize training curves

### Local Training

#### Basic MT10 Training
```bash
python experiments/run_experiment.py \
    --benchmark MT10 \
    --num-epochs 100 \
    --output-dir outputs/mt10_run \
    --seed 42
```

#### MT50 Training
```bash
python experiments/run_experiment.py \
    --benchmark MT50 \
    --num-epochs 200 \
    --output-dir outputs/mt50_run \
    --seed 42
```

### Configuration Options

- `--benchmark`: Choose between MT10 or MT50
- `--num-epochs`: Number of meta-training epochs
- `--output-dir`: Directory for logs and checkpoints
- `--seed`: Random seed for reproducibility
- `--inner-lr`: Inner loop learning rate (default: 0.01)
- `--outer-lr`: Outer loop learning rate (default: 0.001)
- `--num-inner-steps`: Number of gradient steps per task (default: 5)

## Results

See [RESULTS.md](RESULTS.md) for detailed experimental results, including:
- Learning curves across different benchmarks
- Comparison with baselines
- Ablation studies
- Hyperparameter sensitivity analysis

## Project Structure Details

### `meta_learners/maml_ppo.py`
Core MAML+PPO implementation:
- `MAMLPPO` class: Main meta-learner
- Inner/outer loop training logic
- Task sampling and adaptation

### `multi_agent/metaworld_wrapper.py`
Meta-World environment wrapper:
- Standardized interface for MT10/MT50
- Task distribution management
- Observation/action space handling

### `ssm_policies/ssm_actor_critic.py`
SSM-based policy networks:
- Actor network with SSM layers
- Critic network for value estimation
- Shared feature extraction

### `experiments/run_experiment.py`
Main training script:
- Argument parsing
- Environment setup
- Training loop
- Logging and checkpointing

## Dependencies

Key dependencies (see `requirements.txt` for full list):
- PyTorch 2.0+
- Gymnasium 0.29.1
- MuJoCo 3.1.6
- Meta-World (from Farama-Foundation)
- Stable-Baselines3
- Mamba-SSM
- TensorBoard/Weights & Biases

## Troubleshooting

### MuJoCo Installation
If you encounter MuJoCo-related errors:
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install -y libosmesa6-dev patchelf

# Reinstall MuJoCo
pip uninstall mujoco -y
pip install mujoco==3.1.6
```

### MetaWorld Issues
The project uses MetaWorld from the Farama-Foundation GitHub repo (not PyPI 2.0.0 which was yanked):
```bash
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master
```

### GPU/CUDA
For CUDA issues:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with specific CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{metarl-agent-emergence,
  author = {Sunghun Kwag},
  title = {MetaRL-Agent-Emergence: Multi-Agent Meta-RL Framework},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/sunghunkwag/MetaRL-Agent-Emergence}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Meta-World benchmark from Farama Foundation
- MAML implementation inspired by original paper (Finn et al., 2017)
- SSM architectures based on Mamba (Gu & Dao, 2023)
- PPO implementation references Stable-Baselines3

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: sunghunkwag@gmail.com
