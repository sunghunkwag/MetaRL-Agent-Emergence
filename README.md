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
├── test_installation.py    # Installation verification script
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
pip install metaworld

# Verify installation
python test_installation.py
```

## Compatibility and System Requirements

### Tested Configurations

- **Python Versions**: 3.8, 3.9, 3.10, 3.11
- **PyTorch Versions**: 2.0+, 2.1+, 2.2+
- **Operating Systems**: 
  - Ubuntu 20.04+, 22.04+
  - macOS 11+ (Big Sur and later)
  - Windows 10/11 (with WSL2 recommended)

### Hardware Recommendations

- **Minimum**: 8GB RAM, CPU-only
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 6GB+ VRAM
- **Optimal**: 32GB+ RAM, NVIDIA GPU with 12GB+ VRAM (RTX 3080/A5000 or better)

### Known Compatible Environments

- **CUDA**: 11.7, 11.8, 12.0, 12.1
- **cuDNN**: 8.5+
- **Gym**: 0.21.0 - 0.26.0
- **Meta-World**: 2.0+

## Usage

### Running Experiments

```bash
# Basic experiment on MT10
python experiments/run_experiment.py \
  --env metaworld \
  --env_name MT10 \
  --num_tasks 10 \
  --num_epochs 100

# With custom hyperparameters
python experiments/run_experiment.py \
  --env metaworld \
  --env_name MT10 \
  --num_tasks 10 \
  --num_epochs 100 \
  --inner_lr 0.01 \
  --meta_lr 0.001 \
  --batch_size 20

# Using SSM policies
python experiments/run_experiment.py \
  --env metaworld \
  --env_name MT10 \
  --policy_type ssm \
  --num_epochs 100
```

### Expected Performance

On MT10 with 100 epochs:
- Training return: Increases from ~0 to 50-100+
- Test return: Demonstrates generalization to unseen tasks
- Meta-loss: Decreases and stabilizes over training

## Error Handling and Robustness

This framework includes comprehensive error handling mechanisms:

### Automatic Validation

- **Parameter Validation**: All hyperparameters are checked for valid ranges
- **Type Checking**: Input types are validated before processing
- **NaN Detection**: Automatic detection and handling of NaN values in gradients and losses
- **Boundary Checks**: Array indices and dimensions are validated
- **Device Compatibility**: Automatic CPU/GPU detection and fallback

### Error Recovery

- **Graceful Degradation**: System falls back to safe defaults on errors
- **Checkpoint Recovery**: Automatic saving and loading of training checkpoints
- **Memory Management**: Automatic cleanup on out-of-memory errors
- **Timeout Handling**: Environment timeouts are handled gracefully

### Logging and Debugging

- **Comprehensive Logging**: All major operations are logged with timestamps
- **Warning Messages**: Clear warnings for potential issues
- **Error Traces**: Detailed stack traces for debugging
- **Performance Monitoring**: Memory and GPU usage tracking

## Frequently Asked Questions (FAQ)

### Installation Issues

**Q: I'm getting "No module named 'metaworld'" error**

A: Install Meta-World manually:
```bash
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
```

**Q: CUDA out of memory errors**

A: Try these solutions:
1. Reduce batch size: `--batch_size 10`
2. Reduce number of parallel tasks: `--num_tasks 5`
3. Use gradient accumulation: `--gradient_accumulation_steps 2`
4. Enable mixed precision: `--use_amp`

**Q: Installation fails on Apple Silicon (M1/M2/M3)**

A: Use native ARM64 Python and install with:
```bash
arch -arm64 pip install -r requirements.txt
```

### Training Issues

**Q: Training loss is NaN**

A: The framework automatically handles this with:
- Gradient clipping (default: 0.5)
- Learning rate reduction
- Checkpoint rollback to last stable state

You can also manually adjust:
```bash
python experiments/run_experiment.py --grad_clip 0.3 --inner_lr 0.005
```

**Q: Very slow training on CPU**

A: Expected behavior. For faster training:
- Use GPU if available
- Reduce number of inner loop steps: `--num_inner_steps 1`
- Use smaller networks: `--hidden_size 128`

**Q: Out of memory during meta-update**

A: Enable first-order MAML approximation:
```bash
python experiments/run_experiment.py --first_order
```

### Environment Issues

**Q: Meta-World tasks fail to initialize**

A: Check your Meta-World version:
```bash
pip install --upgrade metaworld
```

**Q: Rendering issues on headless servers**

A: Set environment variable:
```bash
export MUJOCO_GL=egl  # or osmesa
```

### Results and Performance

**Q: Poor performance on MT50**

A: MT50 is very challenging. Try:
1. Increase training epochs: `--num_epochs 200`
2. Use more adaptation steps: `--num_inner_steps 3`
3. Tune learning rates: `--inner_lr 0.02 --meta_lr 0.0005`

**Q: How to resume interrupted training?**

A: Training automatically saves checkpoints:
```bash
python experiments/run_experiment.py --resume --checkpoint_path ./checkpoints/latest.pt
```

## Known Issues and Limitations

### Current Limitations

1. **Memory Usage**: Meta-learning requires significant memory for storing task-specific gradients
   - *Workaround*: Use first-order MAML or reduce batch size

2. **Multi-GPU Support**: Currently single GPU only
   - *Status*: Multi-GPU support planned for future release

3. **Windows Native**: Some Meta-World tasks may not work on native Windows
   - *Workaround*: Use WSL2 or Docker

4. **Apple Silicon GPU**: PyTorch MPS backend not fully supported
   - *Workaround*: Use CPU mode on M1/M2/M3 Macs

### Known Bugs

1. **Rare Gradient Explosion**: In very long episodes (>2000 steps)
   - *Workaround*: Set `--max_episode_length 1000`
   - *Status*: Fix in progress

2. **Checkpoint Loading on Different Hardware**: Models trained on GPU may not load on CPU
   - *Workaround*: Use `--device cpu` flag and map location
   - *Status*: Will be fixed in v1.1

3. **TensorBoard Logging Conflicts**: Multiple runs may conflict
   - *Workaround*: Use unique run names: `--run_name experiment_001`

### Reporting Issues

If you encounter a bug:
1. Check if it's a known issue above
2. Run `python test_installation.py` to verify setup
3. Open a GitHub issue with:
   - Python version
   - PyTorch version
   - Error message and stack trace
   - Minimal reproduction code

## Environment-Specific Tips

### Ubuntu/Linux

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-dev libosmesa6-dev libgl1-mesa-glx libglfw3

# For GPU support
sudo apt-get install -y nvidia-cuda-toolkit
```

### macOS

```bash
# Install Homebrew dependencies
brew install glfw3

# Use native Python (not x86 emulation)
which python  # Should show /opt/homebrew/... on M1/M2/M3
```

### Windows (WSL2 Recommended)

```bash
# In WSL2 Ubuntu
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# Follow Ubuntu instructions above
```

### Docker

```bash
# Build image
docker build -t metarl-agent .

# Run with GPU
docker run --gpus all -it metarl-agent

# Run experiments
docker run --gpus all metarl-agent python experiments/run_experiment.py
```

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

## Testing

### Run All Tests

```bash
# Installation verification
python test_installation.py

# Unit tests (if available)
python -m pytest tests/

# Integration test
python experiments/run_experiment.py --num_epochs 2 --num_tasks 2 --test_mode
```

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

**Status**: ✅ Complete runnable framework with SOTA benchmarks and comprehensive error handling
