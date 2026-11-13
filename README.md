# Ivan's DQN Hyperparameter Experiments (Task 3)

Ten DQN experiments on the Atari game **Q\*bert** (`ALE/Qbert-v5`). Each experiment trains a new agent for **100,000 steps** with the same seed (42), the same network and the same game. Each one changes **one** hyperparameter from the baseline (EXP01), so we can see what that change does.

---

## 1. Hyperparameter experiments

### Settings

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | What changed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TASK3_EXP01 | 0.0001 | 0.99 | 32 | 1.0 | 0.1 | 100,000 | Baseline |
| TASK3_EXP02 | 0.00005 | 0.99 | 32 | 1.0 | 0.1 | 100,000 | Smaller learning rate |
| TASK3_EXP03 | 0.00025 | 0.99 | 32 | 1.0 | 0.1 | 100,000 | Bigger learning rate |
| TASK3_EXP04 | 0.0001 | **0.95** | 32 | 1.0 | 0.1 | 100,000 | Smaller gamma |
| TASK3_EXP05 | 0.0001 | **0.995** | 32 | 1.0 | 0.1 | 100,000 | Bigger gamma |
| TASK3_EXP06 | 0.0001 | 0.99 | **64** | 1.0 | 0.1 | 100,000 | Bigger batch |
| TASK3_EXP07 | 0.0001 | 0.99 | **16** | 1.0 | 0.1 | 100,000 | Smaller batch |
| TASK3_EXP08 | 0.0001 | 0.99 | 32 | **0.9** | 0.1 | 100,000 | Lower epsilon start |
| TASK3_EXP09 | 0.0001 | 0.99 | 32 | 1.0 | **0.05** | 100,000 | Lower epsilon end |
| TASK3_EXP10 | 0.0001 | 0.99 | 32 | 1.0 | 0.1 | **50,000** | Faster epsilon decay |

The settings are stored in [`ivan_experiments.json`](ivan_experiments.json).

### Results

"Training" is the mean reward of the last 10 training episodes (noisy, because the agent still makes some random moves). "Evaluation" is the mean reward over the same 10 games (seeds 1000–1009) with no random moves, so it is the fairer comparison.

| Rank (by evaluation) | Experiment | Training reward | Evaluation reward | Evaluation std |
| --- | --- | --- | --- | --- |
| 1 | TASK3_EXP04 | 302.5 | **485.0** | 179.3 |
| 2 | TASK3_EXP10 | 325.0 | 327.5 | 73.7 |
| 3 | TASK3_EXP01 | 207.5 | 300.0 | 123.0 |
| 4 | TASK3_EXP06 | 355.0 | 292.5 | 38.8 |
| 5 | TASK3_EXP05 | 285.0 | 275.0 | 150.0 |
| 6 | TASK3_EXP03 | 315.0 | 270.0 | 56.8 |
| 7 | TASK3_EXP08 | 295.0 | 210.0 | 56.1 |
| 8 | TASK3_EXP09 | 322.5 | 172.5 | 72.8 |
| 9 | TASK3_EXP02 | 377.5 | 145.0 | 95.4 |
| 10 | TASK3_EXP07 | 292.5 | 142.5 | 29.7 |

**In short:** EXP04 (gamma 0.95) scored highest in evaluation. Each setting was trained once for 100,000 steps and the scores change a lot from game to game, so only EXP04's lead is large enough to take seriously. Written notes for every experiment are in [`IvanHyperparameter_experiment_results.md`](IvanHyperparameter_experiment_results.md).

### Plots

- One training plot per experiment (reward, episode length, loss): `plots/ivan/TASK3_EXPxx.png`
- Plots comparing all ten experiments: `plots/ivan/ivan_training_rewards_comparison.png`, `ivan_evaluation_rewards_comparison.png`, `ivan_training_vs_evaluation.png` and `ivan_training_curves.png`

---

## 2. Folder structure

```
.
├── train.py                      training loop
├── play.py                       watch a trained model play
├── run_experiments.py            runs the ten experiments, builds the results table
├── evaluate_experiments.py       tests all ten saved models on the same games
├── record_demo.py                records the gameplay video and GIF
├── dqn.py, dqn_agent.py,         shared network, learning logic, memory and game setup
│   replay_buffer.py, environment.py
├── ivan_experiments.json         the ten experiment settings
├── ivan_observations.json        written notes for each experiment
├── IvanHyperparameter_experiment_results.md   results table and interpretation
├── Ivan_DQN_Atari_Experiments.ipynb           Colab / Jupyter notebook of the whole workflow
├── INSTRUCTIONS.md, PROJECT_GUIDE.md          run instructions and team rules
├── models/ivan/                  final trained model for each experiment (10 files)
├── plots/ivan/                   training plots and comparison plots
├── results/ivan/
│   ├── TASK3_EXPxx/              episodes.csv, training_updates.csv, summary.json
│   ├── evaluation.json           evaluation scores for all ten models
│   └── ivan_*.csv                combined training / evaluation tables
├── demo/                         gameplay video (mp4) and GIF of the best agent
└── pyproject.toml, uv.lock       dependencies (managed with uv)
```

---

## 3. Model checkpoints

| Kind | Where | In GitHub? |
| --- | --- | --- |
| **Final models** (weights after 100,000 steps) | `models/ivan/TASK3_EXP01.pth` to `TASK3_EXP10.pth` (about 6.7 MB each) | Yes |
| **Intermediate checkpoints** (saved every 10,000 steps) | `results/ivan/TASK3_EXPxx/checkpoint_step_10000.pth` to `checkpoint_step_100000.pth` (100 files, about 690 MB in total) | **No**, they are too large |

The intermediate checkpoints stay on my computer and are not uploaded. Anyone can recreate them by running the experiment again with the same settings, because `train.py` saves a checkpoint every 10,000 steps.

To use a saved final model:

```bash
uv run python play.py --model models/ivan/TASK3_EXP04.pth --episodes 3
```

`TASK3_EXP04.pth` is the best model by evaluation.
