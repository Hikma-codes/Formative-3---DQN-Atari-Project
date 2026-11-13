# Ivan — Task 3: Individual Hyperparameter Experiments

Environment: `ALE/Qbert-v5`. All results must come from actual runs. No rewards are pre-filled.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Observed behaviour |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TASK3_EXP01 | 0.0001 | 0.99 | 32 | 1.0 | 0.1 | 100000 | Completed 276 episodes in 100000 steps; mean reward (last 10): 207.50; mean episode length: 352.1; optimizer updates: 24501. Baseline. Reward rose from about 165 (first 50 games) to about 359 (last 50). Its last-10 number (207.5) is the lowest of all runs, but the last-50 average is among the highest, so last-10 is just noisy. Greedy evaluation: 300.0, middle of the pack. |
| TASK3_EXP02 | 5e-05 | 0.99 | 32 | 1.0 | 0.1 | 100000 | Completed 285 episodes in 100000 steps; mean reward (last 10): 377.50; mean episode length: 408.0; optimizer updates: 24501. Smaller learning rate (5e-5). Best last-10 training reward (377.5) but one of the worst greedy scores (145.0). The training number looked good because of a lucky stretch, not because the agent was better. Learning was slower and gentler (final loss 0.0138). |
| TASK3_EXP03 | 0.00025 | 0.99 | 32 | 1.0 | 0.1 | 100000 | Completed 279 episodes in 100000 steps; mean reward (last 10): 315.00; mean episode length: 392.6; optimizer updates: 24501. Bigger learning rate (2.5e-4). Highest final loss of all runs (0.0351), so updates were rougher. Greedy score 270.0 with a fairly small spread (std 57), so it played consistently but not well. |
| TASK3_EXP04 | 0.0001 | 0.95 | 32 | 1.0 | 0.1 | 100000 | Completed 279 episodes in 100000 steps; mean reward (last 10): 302.50; mean episode length: 381.4; optimizer updates: 24501. Gamma 0.95 (looks less far ahead). Best greedy score by a clear margin: 485.0, with games from 125 to 675. It also had the lowest final loss (0.0092), partly because a smaller gamma makes the targets smaller. The spread is large (std 179), so the lead is real but not huge. |
| TASK3_EXP05 | 0.0001 | 0.995 | 32 | 1.0 | 0.1 | 100000 | Completed 279 episodes in 100000 steps; mean reward (last 10): 285.00; mean episode length: 372.0; optimizer updates: 24501. Gamma 0.995 (looks very far ahead). Training reward was good early (first 50 games: 189) but flattened later (last 50: 324). Greedy score 275.0 with a wide spread (std 150). Higher loss (0.0288) than the baseline, which fits with larger, harder-to-predict targets. |
| TASK3_EXP06 | 0.0001 | 0.99 | 64 | 1.0 | 0.1 | 100000 | Completed 274 episodes in 100000 steps; mean reward (last 10): 355.00; mean episode length: 412.9; optimizer updates: 24501. Batch size 64. Best mid-training reward (298) and a steady greedy score of 292.5 with the smallest spread among the stronger runs (std 39). Bigger batches seem to give more stable play, though the average is not higher than the baseline. |
| TASK3_EXP07 | 0.0001 | 0.99 | 16 | 1.0 | 0.1 | 100000 | Completed 291 episodes in 100000 steps; mean reward (last 10): 292.50; mean episode length: 352.7; optimizer updates: 24501. Batch size 16. Lowest last-50 training reward (289) and the lowest greedy score (142.5). Small spread (std 30), so it was consistently weak, not just unlucky. Smaller batches gave noisier learning here. |
| TASK3_EXP08 | 0.0001 | 0.99 | 32 | 0.9 | 0.1 | 100000 | Completed 285 episodes in 100000 steps; mean reward (last 10): 295.00; mean episode length: 362.3; optimizer updates: 24501. Epsilon start 0.9 instead of 1.0. Almost the same as the baseline: last-50 reward 359 and greedy score 210.0. Starting with a few fewer random moves made no clear difference. |
| TASK3_EXP09 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Completed 277 episodes in 100000 steps; mean reward (last 10): 322.50; mean episode length: 403.9; optimizer updates: 24501. Epsilon end 0.05 (less random at the end). Produced the single best training game of all runs (1325), but the greedy score was low (172.5). One great game did not mean a better average. |
| TASK3_EXP10 | 0.0001 | 0.99 | 32 | 1.0 | 0.1 | 50000 | Completed 272 episodes in 100000 steps; mean reward (last 10): 325.00; mean episode length: 406.2; optimizer updates: 24501. Epsilon decay over 50,000 steps (stops exploring sooner). Second-best greedy score (327.5) with a modest spread (std 74). Exploring for less time did not hurt in this short training budget, and may have helped. |

## Interpretation

Compare runs using the same total environment-step budget and the *last-10 completed episode mean raw reward* (a noisy training diagnostic). For a defensible best/worst claim, evaluate saved models greedily over multiple seeded episodes with a consistent protocol; training rewards alone are not sufficient.

Current highest training reward: `TASK3_EXP02` (377.50).
Current lowest training reward: `TASK3_EXP01` (207.50).

These are training diagnostics, not controlled evaluation results.

### Written interpretation

All ten agents got better during training: the average reward over the first 50 games was roughly 140 to 190, and over the last 50 games roughly 290 to 360. But training reward is a poor way to rank the runs, because the agent was still making random moves (epsilon fell from 1.0 to 0.1 over the whole run) and the last-10 average jumps around. When every saved model played the same 10 games using only its best move, the picture changed: EXP04 (gamma 0.95) was best at 485.0, EXP10 (shorter exploration) and EXP01 (baseline) were next at 327.5 and 300.0, and EXP07 (batch size 16) and EXP02 (smaller learning rate) were worst at 142.5 and 145.0. Be careful with these conclusions. Each setting was trained once, with one seed, for only 100,000 steps, and the scores vary a lot from game to game (standard deviation 30 to 179). Only EXP04's lead is large enough to take seriously; differences among the middle runs (about 270 to 330) are within the noise. Loss values should only be compared between runs with the same gamma, since gamma changes the size of the targets. A fairer test would repeat each setting with several seeds and train for longer.

## Reproducibility

Every experiment saves its exact configuration to `results/ivan/TASK3_EXPxx/summary.json`, logs episodes to `episodes.csv`, optimization updates to `training_updates.csv`, plots to `plots/ivan/`, and final weights to `models/ivan/`.

## Greedy evaluation

Each saved model played the same 10 seeded episodes (seeds 1000-1009) with greedy actions (epsilon = 0).

| Rank | Experiment | Mean reward | Std | Min | Max |
| --- | --- | --- | --- | --- | --- |
| 1 | TASK3_EXP04 | 485.0 | 179.3 | 125 | 675 |
| 2 | TASK3_EXP10 | 327.5 | 73.7 | 125 | 400 |
| 3 | TASK3_EXP01 | 300.0 | 123.0 | 200 | 625 |
| 4 | TASK3_EXP06 | 292.5 | 38.8 | 200 | 350 |
| 5 | TASK3_EXP05 | 275.0 | 150.0 | 175 | 600 |
| 6 | TASK3_EXP03 | 270.0 | 56.8 | 200 | 400 |
| 7 | TASK3_EXP08 | 210.0 | 56.1 | 175 | 325 |
| 8 | TASK3_EXP09 | 172.5 | 72.8 | 75 | 275 |
| 9 | TASK3_EXP02 | 145.0 | 95.4 | 75 | 425 |
| 10 | TASK3_EXP07 | 142.5 | 29.7 | 125 | 200 |

Highest greedy-evaluation reward: `TASK3_EXP04` (485.0). Lowest: `TASK3_EXP07` (142.5).
