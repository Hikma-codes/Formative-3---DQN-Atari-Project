# DQN Atari Project – Shared Project Guide

## 1. Project Overview

Our group is implementing a Deep Q-Network (DQN) agent from scratch using PyTorch and Gymnasium.

The selected Atari game is Q*bert.

We will implement the neural network, experience replay, epsilon-greedy action selection, target network, Bellman equation updates, training and evaluation ourselves.

Ready-made DQN implementations and reinforcement learning libraries such as Stable-Baselines3, RLlib, CleanRL and TorchRL are not permitted.

## 2. Shared Environment

- Game: Q*bert
- Environment ID: `ALE/Qbert-v5`
- Framework: PyTorch
- Environment library: Gymnasium
- Emulator: Arcade Learning Environment (ALE)
- Algorithm: Deep Q-Network

## 3. Shared Observation Configuration

- Image size: 84 × 84 pixels
- Image format: Grayscale
- Frame stack: 4 consecutive frames
- Frame skip: 4
- Expected observation shape: `(4, 84, 84)`
- Observation data type: `uint8`

All members must use the same preprocessing configuration.

## 4. Shared Files and Responsibilities

| File | Purpose |
|---|---|
| `environment.py` | Creates and preprocesses Q*bert |
| `dqn.py` | Defines the CNN architecture |
| `replay_buffer.py` | Stores and samples experiences |
| `dqn_agent.py` | Epsilon schedule, action selection, Bellman update and target network sync |
| `test_learning.py` | Tests the replay buffer and learning logic |
| `train.py` | Trains the DQN agent |
| `play.py` | Loads and demonstrates trained models |
| `test_environment.py` | Tests the Atari environment |
| `test_dqn.py` | Tests the CNN |
| `PROJECT_GUIDE.md` | Shared naming and implementation guide |
| `README.md` | Main project documentation |

## 5. Naming Conventions

### Classes
- `DQN`
- `ReplayBuffer`

### Functions
- `make_env()`
- `select_action()`
- `get_epsilon()`
- `train_step()`
- `save_model()`
- `load_model()`
- `evaluate_agent()`
- `update_target_network()`
- `get_device()`

### Shared variables
- `ENV_NAME`
- `online_network`
- `target_network`
- `optimizer`
- `learning_rate`
- `gamma`
- `batch_size`
- `epsilon_start`
- `epsilon_end`
- `epsilon_decay`
- `total_steps`
- `episode_reward`
- `episode_length`
- `state`
- `next_state`
- `action`
- `reward`
- `done`

## 6. CNN Architecture

The shared DQN architecture uses:

1. First convolutional layer: 32 filters, kernel size 8, stride 4.
2. Second convolutional layer: 64 filters, kernel size 4, stride 2.
3. Third convolutional layer: 64 filters, kernel size 3, stride 1.
4. Fully connected layer: 512 units.
5. Output layer: 6 Q-values, one per available action.

The network receives four stacked 84 × 84 grayscale frames.

## 7. Repository Structure

```text
Formative-3---DQN-Atari-Project/
├── environment.py
├── dqn.py
├── replay_buffer.py
├── dqn_agent.py
├── train.py
├── play.py
├── test_environment.py
├── test_dqn.py
├── test_learning.py
├── models/
│   ├── member_1/
│   ├── member_2/
│   ├── member_3/
│   └── member_4/
├── results/
│   ├── member_1/
│   ├── member_2/
│   ├── member_3/
│   └── member_4/
├── plots/
├── docs/
├── README.md
├── PROJECT_GUIDE.md
├── pyproject.toml
└── uv.lock
```

## 8. Experiment Naming

Each member completes 10 hyperparameter experiments.

Naming format:

- `M1_EXP01` through `M1_EXP10`
- `M2_EXP01` through `M2_EXP10`
- `M3_EXP01` through `M3_EXP10`
- `M4_EXP01` through `M4_EXP10`

Example checkpoint:

`models/member_1/M1_EXP01.pth`

Each experiment must record its hyperparameters, training budget, performance and observations.

## 9. Team Responsibilities

- Member 1: Environment, preprocessing and CNN architecture.
- Member 2: Replay buffer, epsilon-greedy, Bellman update, loss and target network.
- Member 3: Training pipeline, integration, logging and checkpoint saving.
- Member 4: Evaluation, gameplay demonstration, plots and documentation.

All members must understand the overall DQN implementation and complete their own 10 experiments.

## 10. Collaboration Rules

1. Use `uv` for all dependency management.
2. Do not use `pip` or another package manager.
3. Keep the same environment and preprocessing settings.
4. Do not change shared function or class interfaces without communicating with the team.
5. Use the same CNN architecture for comparable experiments.
6. Record actual experimental results.
7. Keep individual experiment files and checkpoints organised.
8. Commit work regularly.
9. Test code before handing it to another member.
10. All members must understand how the components fit together.