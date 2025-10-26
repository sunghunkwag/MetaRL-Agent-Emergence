# Meta-Learners Module

This directory contains meta-learning algorithms for multi-agent reinforcement learning.

## Overview

Meta-learning enables agents to quickly adapt to new tasks by learning how to learn. This module implements state-of-the-art meta-learning algorithms optimized for multi-agent scenarios.

## Implemented Algorithms

### MAML (Model-Agnostic Meta-Learning)
- **File**: `maml.py`
- **Description**: Gradient-based meta-learning that learns initialization parameters enabling fast adaptation
- **Key Features**:
  - First-order and second-order variants
  - Support for multi-agent coordination
  - Efficient task adaptation through few gradient steps

### Reptile
- **File**: `maml.py` (ReptileMetaLearner class)
- **Description**: Simplified first-order meta-learning algorithm
- **Key Features**:
  - More memory-efficient than MAML
  - Competitive performance with simpler implementation

## Usage Example

```python
from meta_learners.maml import MAMLMetaLearner
from ssm_policies.ssm_policy import SSMPolicy

# Initialize policy network
policy = SSMPolicy(state_dim=64, action_dim=4, hidden_dim=256)

# Create MAML meta-learner
maml = MAMLMetaLearner(
    policy_network=policy,
    meta_lr=1e-3,
    inner_lr=1e-2,
    num_inner_steps=5
)

# Meta-training loop
for task_batch in task_distribution:
    meta_loss = maml.meta_update(task_batch)
    print(f"Meta-loss: {meta_loss}")

# Adapt to new task
adapted_policy = maml.adapt_to_new_task(new_task_data)
```

## Future Additions

- [ ] Meta-RL specific algorithms (RL²)
- [ ] Context-based meta-learning
- [ ] Multi-task learning variants
- [ ] Hierarchical meta-learning
