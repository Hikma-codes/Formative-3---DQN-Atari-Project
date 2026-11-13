# Silver's DQN Hyperparameter Experiments

This folder contains the evidence from Silver's (Nshuti Shalom Silver Jr) ten DQN hyperparameter experiments on Gymnasium `ALE/Qbert-v5`.

## Experiments

- 10 distinct hyperparameter configurations.
- 100,000 training steps per experiment, seed 42.
- Training performed on Google Colab with a Tesla T4 GPU.
- 10 greedy evaluation episodes per trained model (seeds 42 to 51), the same protocol as Hikma's experiments.
- An extended evaluation of 60 episodes per model (seeds 42 to 101).

Best configuration: `SILVER_EXP10` (learning rate 0.0002, gamma 0.97, batch size 256).

The experiment table, rankings and interpretation are in `results/SilverHyperparameter_experiment_results.md`.

## Folder structure

- `configs/`: the ten experiment configurations.
- `results/`: experiment report, training and evaluation results, and the individual experiment records.
- `plots/`: per-experiment training plots, learning curves and comparison plots.
- `models/`: the final trained model of each experiment (`SILVER_EXP01.pth` to `SILVER_EXP10.pth`).

## Model checkpoints

The ten final models are stored in `models/` because the assignment requires each member's models in the repository. The intermediate training checkpoints are kept on Google Drive only.
