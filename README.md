# Q*bert Deep Q-Network (DQN)

This project trains a custom convolutional DQN agent to play Q*bert (`ALE/Qbert-v5`) using Gymnasium and PyTorch. The CNN, replay buffer, action selection, and optimization are implemented by our group, without ready-made reinforcement learning agents.

## Contributions

- Task 1: Atari environment preprocessing and CNN (`environment.py`, `dqn.py`).
- Task 2: Replay buffer and DQN updates (`replay_buffer.py`, `dqn_agent.py`).
- **Task 3 — Ivan:** Training pipeline (`train.py`), ten hyperparameter experiments (`ivan_experiments.json`, `run_experiments.py`), model checkpoints, metrics and plots.
- Task 4: Greedy gameplay demonstration (`play.py`); integration copy included for testing with the same DQN architecture.

Confirm the other group members' names and responsibilities with your team before final submission.

## Architecture

```text
4 grayscale frames (4 × 84 × 84)
        ↓
Conv2d(4→32, 8×8, stride 4) + ReLU
        ↓
Conv2d(32→64, 4×4, stride 2) + ReLU
        ↓
Conv2d(64→64, 3×3, stride 1) + ReLU
        ↓
Flatten → Linear(3136→512) + ReLU
        ↓
Linear(512→number of actions) → Q-values
```

## How training works

The agent chooses random actions with probability epsilon and otherwise selects the action with the highest estimated Q-value. It stores game transitions in a replay buffer and samples random mini-batches. The online network predicts Q(s,a), and the target network supplies a stable Bellman target, `reward + gamma * max(next Q) * (1 - done)`. Huber loss and Adam update the online network. Target weights are copied from the online network periodically.

## Run using uv

```bash
uv sync
uv run python test_dqn.py
uv run python test_learning.py
uv run python run_experiments.py --list
uv run python run_experiments.py --experiment TASK3_EXP01
uv run python run_experiments.py --all
```

For a quick pipeline test, see `INSTRUCTIONS.md`. Training ten meaningful Atari experiments requires significant compute time; the configurations are **planned**, not measured results.

Each experiment writes `models/ivan/TASK3_EXPxx.pth`, a CSV training log, a JSON summary, and `plots/ivan/TASK3_EXPxx.png`. The table `IvanHyperparameter_experiment_results.md` updates after successful runs; explain observations in your own words after reviewing the results.

## Live gameplay

After selecting the experiment that performs best in repeated evaluation:

```bash
uv run python run_experiments.py --export-best TASK3_EXP01
uv run python play.py --model dqn_model.pth --episodes 3
```

To compare models without a display, use `uv run python play.py --model models/ivan/TASK3_EXP01.pth --episodes 5 --no-render`. Greedy evaluation rewards are different from training-time epsilon-greedy rewards.

## Submission status

The training code and experiment plan are included. **Real trained models, reward plots, completed observations, the other members' result tables, and the task allocation PDF must still be produced or added.** No model performance is claimed before actual game training and evaluation.
