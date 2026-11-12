# Kalisa — Task 3: Training and Hyperparameter Experiments

This contribution builds on the shared `environment.py`, `dqn.py`, `replay_buffer.py`, and `dqn_agent.py` without changing their interfaces.

## Setup and checks

Use **uv**, not pip:

```bash
uv sync
uv run python test_dqn.py
uv run python test_learning.py
uv run python test_environment.py
uv run python run_experiments.py --list
```

`test_environment.py` may require a graphical display. For a short real training check:

```bash
uv run python train.py --experiment-id TASK3_SMOKE --max-steps 100 --learning-starts 8 --replay-capacity 64 --batch-size 4 --train-frequency 4 --target-update-frequency 20 --checkpoint-frequency 100 --log-frequency 20 --device cpu
```

Check `results/kalisa/TASK3_SMOKE/summary.json` for `optimizer_updates > 0`. This is a pipeline check, **not evidence that the agent has learned to play**.

## Ten experiments

The six required hyperparameters are recorded in `kalisa_experiments.json`. Each experiment starts with a new DQN model, optimizer, buffer, and seeded environment. The default training budget is 100,000 agent decisions per experiment. For stronger performance, choose an equal, larger budget for all ten runs.

```bash
uv run python run_experiments.py --experiment TASK3_EXP01
uv run python run_experiments.py --all
uv run python run_experiments.py --all --max-steps 500000
```

The runner **skips previously completed experiment IDs** to avoid deleting results. If you change the budget, move or archive the previous results and model before rerunning with that ID; do not compare unequal budgets as if they were the same experiment.

For each experiment, the code produces:

- `models/kalisa/TASK3_EXPxx.pth`: model weights.
- `results/kalisa/TASK3_EXPxx/episodes.csv`: rewards, episode lengths, epsilon, steps and mean loss.
- `results/kalisa/TASK3_EXPxx/training_updates.csv`: optimization loss and mean predicted Q.
- `results/kalisa/TASK3_EXPxx/summary.json`: exact configuration and numerical training summary.
- `plots/kalisa/TASK3_EXPxx.png`: reward, episode length and loss.
- `KalisaHyperparameter_experiment_results.md`: numerical results table, plus room for personal interpretation.

## Evaluate and demonstrate

```bash
uv run python play.py --model models/kalisa/TASK3_EXP01.pth --episodes 5 --no-render
uv run python run_experiments.py --export-best TASK3_EXP01
uv run python play.py --model dqn_model.pth
```

Choose the best model based on repeated greedy evaluation with consistent seeds, not only training-time rewards. The last command opens live gameplay and requires a desktop display.

## Explain this during Q&A

- **DQN:** approximates the expected future discounted reward of each action with a CNN.
- **Replay buffer:** randomizes past transitions so nearby correlated game frames do not dominate a mini-batch.
- **Online vs target network:** the online network learns; the target network supplies stable Bellman targets and is periodically synchronized.
- **Epsilon-greedy:** balances exploring random actions and exploiting Q-value predictions. Epsilon decays over environment steps.
- **Gamma:** controls how strongly the agent values future rewards.
- **Learning rate / batch size:** control update size and how many experiences contribute to one update.
- **Huber loss:** compares predicted Q-values with the Bellman target, limiting the effect of very large errors.
- **Raw vs clipped rewards:** reports use original game rewards, while optimization clips rewards for stability.

## Collaboration and completeness

Only Task 3 training and experiment management files need merging into the group's main repository. `play.py` is a working **integration draft**; coordinate with the teammate responsible for the final shared evaluator before replacing their version. The group still needs real training results, saved weights, completed observations, other members' tables, a final task allocation sheet, and a live demonstration.
