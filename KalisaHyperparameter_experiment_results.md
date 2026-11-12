# Kalisa — Task 3: Individual Hyperparameter Experiments

Environment: `ALE/Qbert-v5`. All results must come from actual runs. No rewards are pre-filled.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Observed behaviour |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TASK3_EXP01 | 0.0001 | 0.99 | 32 | 1.0 | 0.1 | 100000 | Completed 276 episodes in 100000 steps; mean reward (last 10): 207.50; mean episode length: 352.1; optimizer updates: 24501. Add qualitative observations after reviewing gameplay. |
| TASK3_EXP02 | 5e-05 | 0.99 | 32 | 1.0 | 0.1 | 100000 | Completed 285 episodes in 100000 steps; mean reward (last 10): 377.50; mean episode length: 408.0; optimizer updates: 24501. Add qualitative observations after reviewing gameplay. |
| TASK3_EXP03 | 0.00025 | 0.99 | 32 | 1.0 | 0.1 | 100000 | Completed 279 episodes in 100000 steps; mean reward (last 10): 315.00; mean episode length: 392.6; optimizer updates: 24501. Add qualitative observations after reviewing gameplay. |
| TASK3_EXP04 | 0.0001 | 0.95 | 32 | 1.0 | 0.1 | 100000 | Completed 279 episodes in 100000 steps; mean reward (last 10): 302.50; mean episode length: 381.4; optimizer updates: 24501. Add qualitative observations after reviewing gameplay. |
| TASK3_EXP05 | 0.0001 | 0.995 | 32 | 1.0 | 0.1 | 100000 | Completed 279 episodes in 100000 steps; mean reward (last 10): 285.00; mean episode length: 372.0; optimizer updates: 24501. Add qualitative observations after reviewing gameplay. |
| TASK3_EXP06 | 0.0001 | 0.99 | 64 | 1.0 | 0.1 | 100000 | Completed 274 episodes in 100000 steps; mean reward (last 10): 355.00; mean episode length: 412.9; optimizer updates: 24501. Add qualitative observations after reviewing gameplay. |
| TASK3_EXP07 | 0.0001 | 0.99 | 16 | 1.0 | 0.1 | 100000 | Completed 291 episodes in 100000 steps; mean reward (last 10): 292.50; mean episode length: 352.7; optimizer updates: 24501. Add qualitative observations after reviewing gameplay. |
| TASK3_EXP08 | 0.0001 | 0.99 | 32 | 0.9 | 0.1 | 100000 | Completed 285 episodes in 100000 steps; mean reward (last 10): 295.00; mean episode length: 362.3; optimizer updates: 24501. Add qualitative observations after reviewing gameplay. |
| TASK3_EXP09 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Completed 277 episodes in 100000 steps; mean reward (last 10): 322.50; mean episode length: 403.9; optimizer updates: 24501. Add qualitative observations after reviewing gameplay. |
| TASK3_EXP10 | 0.0001 | 0.99 | 32 | 1.0 | 0.1 | 50000 | Completed 272 episodes in 100000 steps; mean reward (last 10): 325.00; mean episode length: 406.2; optimizer updates: 24501. Add qualitative observations after reviewing gameplay. |

## Interpretation

Compare runs using the same total environment-step budget and the *last-10 completed episode mean raw reward* (a noisy training diagnostic). For a defensible best/worst claim, evaluate saved models greedily over multiple seeded episodes with a consistent protocol; training rewards alone are not sufficient.

Current highest training reward: `TASK3_EXP02` (377.50).
Current lowest training reward: `TASK3_EXP01` (207.50).

These are training diagnostics, not controlled evaluation results.

## Reproducibility

Every experiment saves its exact configuration to `results/kalisa/TASK3_EXPxx/summary.json`, logs episodes to `episodes.csv`, optimization updates to `training_updates.csv`, plots to `plots/kalisa/`, and final weights to `models/kalisa/`.
