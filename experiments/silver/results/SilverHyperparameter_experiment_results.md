# Silver: Hyperparameter Experiment Results

Member name: Nshuti Shalom Silver Jr

Environment: `ALE/Qbert-v5`. All numbers below come from actual runs.

## Setup

- Training budget: 100,000 environment steps per experiment, seed 42.
- Training device: cuda (Google Colab).
- Fixed settings: replay capacity 20,000, learning starts at 2,000 steps, one update every 4 steps, target network sync every 2,000 steps, linear epsilon decay, reward clipping on, Double DQN off.
- Baseline: learning rate 0.0001, gamma 0.99, batch size 32, epsilon 1.0 to 0.1 over 100,000 steps.
- Evaluation: each model played 10 episodes (seeds 42 to 51) with greedy actions (epsilon = 0).
- Reference: a random agent on the same episodes scored a mean reward of 155.0 (mean length 327 steps).

## Experiment table

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Observed behaviour |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SILVER_EXP01 | 0.0001 | 0.99 | 32 | 1.0 | 0.1 | 100000 | Baseline (control run). Training: 269 episodes, last-10 mean reward 455.0 at epsilon 0.10. Greedy evaluation: mean reward 345.0 (std 176.3, min 225, max 600), mean episode length 389 steps, rank 4 of 10. Learned steadily and had the highest training reward of the ten (449.0 over the last 50 episodes), but its greedy policy scored either 600 or 225 to 250. |
| SILVER_EXP02 | 0.0005 | 0.99 | 32 | 1.0 | 0.1 | 100000 | High learning rate. Training: 276 episodes, last-10 mean reward 277.5 at epsilon 0.10. Greedy evaluation: mean reward 317.5 (std 59.0, min 250, max 450), mean episode length 597 steps, rank 7 of 10. Lowest training reward of the ten over the last 50 episodes (298.5): the curve stalled around 300 and fell near the end, so this learning rate is too high. |
| SILVER_EXP03 | 0.0001 | 0.9 | 32 | 1.0 | 0.1 | 100000 | Low gamma (short-sighted agent). Training: 280 episodes, last-10 mean reward 335.0 at epsilon 0.10. Greedy evaluation: mean reward 272.5 (std 64.0, min 200, max 425), mean episode length 506 steps, rank 9 of 10. Below the baseline for most of training. Predicted Q-values stayed small (about 0.4 against 2.4 for the baseline) because future rewards are discounted heavily. |
| SILVER_EXP04 | 0.0001 | 0.999 | 32 | 1.0 | 0.1 | 100000 | Very high gamma (far-sighted agent). Training: 275 episodes, last-10 mean reward 410.0 at epsilon 0.10. Greedy evaluation: mean reward 217.5 (std 23.7, min 200, max 250), mean episode length 430 steps, rank 10 of 10. Training reward looked normal, but the greedy policy scored only 200 to 250 in every evaluation episode: the worst model. |
| SILVER_EXP05 | 0.0001 | 0.99 | 256 | 1.0 | 0.1 | 100000 | Large batch size. Training: 267 episodes, last-10 mean reward 430.0 at epsilon 0.10. Greedy evaluation: mean reward 560.0 (std 125.9, min 450, max 850), mean episode length 520 steps, rank 2 of 10. Second-best greedy policy and the most consistent of the top models (lowest score 450). Training took about twice as long (10.3 against 4.8 minutes). |
| SILVER_EXP06 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 100000 | Very low final epsilon. Training: 282 episodes, last-10 mean reward 330.0 at epsilon 0.01. Greedy evaluation: mean reward 312.5 (std 41.2, min 225, max 350), mean episode length 341 steps, rank 8 of 10. Training reward stayed below the baseline for most of the run, and evaluation scores fell in a narrow band (225 to 350). |
| SILVER_EXP07 | 0.0001 | 0.99 | 32 | 1.0 | 0.1 | 25000 | Fast epsilon decay. Training: 267 episodes, last-10 mean reward 395.0 at epsilon 0.10. Greedy evaluation: mean reward 357.5 (std 274.9, min 125, max 750), mean episode length 579 steps, rank 3 of 10. Training reward tracked the baseline. Evaluation was mostly split between high scores (725 to 750) and low ones (125 to 200), giving the largest spread. |
| SILVER_EXP08 | 0.0001 | 0.99 | 32 | 0.5 | 0.1 | 100000 | Low starting epsilon. Training: 279 episodes, last-10 mean reward 452.5 at epsilon 0.10. Greedy evaluation: mean reward 345.0 (std 72.5, min 250, max 475), mean episode length 435 steps, rank 5 of 10. Same evaluation mean as the baseline with less spread, so starting with less exploration made no measurable difference. |
| SILVER_EXP09 | 0.0001 | 0.97 | 256 | 1.0 | 0.1 | 100000 | Large batch + gamma 0.97. Training: 268 episodes, last-10 mean reward 430.0 at epsilon 0.10. Greedy evaluation: mean reward 335.0 (std 24.2, min 300, max 375), mean episode length 468 steps, rank 6 of 10. Training reward matched SILVER_EXP05 (430.0), but greedy evaluation stayed at 300 to 375, so adding gamma 0.97 to the large batch did not help on its own. |
| SILVER_EXP10 | 0.0002 | 0.97 | 256 | 1.0 | 0.1 | 100000 | Large batch + gamma 0.97 + learning rate 2e-4. Training: 269 episodes, last-10 mean reward 410.0 at epsilon 0.10. Greedy evaluation: mean reward 592.5 (std 222.4, min 325, max 900), mean episode length 518 steps, rank 1 of 10. Best greedy policy and the highest single evaluation score (900). Selected as my final configuration. |

## Evaluation ranking

| Rank | Experiment | Evaluation mean reward | Std | Min | Max | Mean episode length | Training last-10 mean reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SILVER_EXP10 | 592.5 | 222.4 | 325 | 900 | 518 | 410.0 |
| 2 | SILVER_EXP05 | 560.0 | 125.9 | 450 | 850 | 520 | 430.0 |
| 3 | SILVER_EXP07 | 357.5 | 274.9 | 125 | 750 | 579 | 395.0 |
| 4 | SILVER_EXP01 | 345.0 | 176.3 | 225 | 600 | 389 | 455.0 |
| 5 | SILVER_EXP08 | 345.0 | 72.5 | 250 | 475 | 435 | 452.5 |
| 6 | SILVER_EXP09 | 335.0 | 24.2 | 300 | 375 | 468 | 430.0 |
| 7 | SILVER_EXP02 | 317.5 | 59.0 | 250 | 450 | 597 | 277.5 |
| 8 | SILVER_EXP06 | 312.5 | 41.2 | 225 | 350 | 341 | 330.0 |
| 9 | SILVER_EXP03 | 272.5 | 64.0 | 200 | 425 | 506 | 335.0 |
| 10 | SILVER_EXP04 | 217.5 | 23.7 | 200 | 250 | 430 | 410.0 |

Best configuration in evaluation: `SILVER_EXP10` (592.5).
Worst configuration in evaluation: `SILVER_EXP04` (217.5).

## Extended evaluation (60 episodes)

Ten episodes are few, and a greedy policy tends to repeat the same few scores, so the saved models were evaluated again on 60 episodes each (seeds 42 to 101) with the same greedy protocol. This evaluation was run on CPU. Its first ten episodes reproduce the Colab results above exactly, score for score and step for step.

| Rank | Experiment | Mean reward (60 episodes) | Standard error | Std | Median | Min | Max | Mean reward (first 10 episodes) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SILVER_EXP10 | 631.2 | 31.2 | 241.4 | 675 | 250 | 1275 | 592.5 |
| 2 | SILVER_EXP05 | 516.2 | 14.6 | 113.0 | 525 | 200 | 850 | 560.0 |
| 3 | SILVER_EXP09 | 378.3 | 12.5 | 96.8 | 350 | 300 | 625 | 335.0 |
| 4 | SILVER_EXP08 | 362.9 | 14.1 | 109.2 | 350 | 125 | 775 | 345.0 |
| 5 | SILVER_EXP01 | 333.8 | 20.2 | 156.6 | 250 | 225 | 600 | 345.0 |
| 6 | SILVER_EXP07 | 331.2 | 30.3 | 235.0 | 200 | 125 | 775 | 357.5 |
| 7 | SILVER_EXP06 | 304.2 | 4.6 | 36.0 | 325 | 225 | 375 | 312.5 |
| 8 | SILVER_EXP02 | 299.6 | 8.4 | 65.2 | 325 | 125 | 450 | 317.5 |
| 9 | SILVER_EXP03 | 285.8 | 11.9 | 92.0 | 250 | 150 | 475 | 272.5 |
| 10 | SILVER_EXP04 | 230.0 | 3.7 | 28.3 | 225 | 150 | 300 | 217.5 |

A random agent on the same 60 episodes scored a mean reward of 150.0.

The ranking of the top two and bottom two models is unchanged. With more episodes, `SILVER_EXP10` (631.2) is clearly ahead of `SILVER_EXP05` (516.2): the gap of 115 is more than three times its standard error (about 34).

## Plots

- `experiments/silver/plots/silver_learning_curves.png`: training reward of each experiment against the baseline.
- `experiments/silver/plots/silver_training_vs_evaluation.png`: training and evaluation reward by experiment.
- `experiments/silver/plots/silver_extended_evaluation.png`: 60-episode evaluation reward by experiment.
- `experiments/silver/plots/SILVER_EXPxx.png`: reward, episode length and loss for each experiment.

## Interpretation

### What improved performance

- **Batch size 256.** The two best greedy policies both used it: `SILVER_EXP10` (631.2 over 60 episodes) and `SILVER_EXP05` (516.2), against 333.8 for the baseline. Each update averages over eight times more experiences, which gives less noisy gradients. The cost is training time: about 10 minutes per run instead of 5.
- **Learning rate 2e-4 together with gamma 0.97 and the large batch** gave the best model. Hikma's best model also used learning rate 2e-4 (585.0 on the same ten evaluation seeds).

Batch size alone is not a guarantee. `SILVER_EXP09` also used batch size 256 and finished near the baseline (378.3), and Hikma's batch size 128 run scored 317.5.

### What harmed performance

- **Gamma 0.999** gave the worst model (230.0, the random agent scores 150.0). The agent has to account for rewards thousands of steps ahead, which is too hard to learn in 100,000 steps.
- **Gamma 0.90** was also below the baseline (285.8). The agent only values the next few jumps. With teammates' results for 0.95 and 0.97, gamma appears to work best around 0.95 to 0.97 and worse at both extremes.
- **Learning rate 5e-4** had the lowest training reward of the ten and a below-baseline evaluation (299.6). The updates are too large for the Q-values to settle.

### What made little difference

The epsilon settings. Fast decay (331.2), a low starting epsilon (362.9) and a very low final epsilon (304.2) all stayed close to the baseline (333.8). In this training budget, how the network learns mattered more than how exploration was scheduled.

### Training reward compared with evaluation reward

Training reward was a poor guide to the quality of the greedy policy. The baseline had the highest training reward (449.0 over its last 50 episodes) but an average evaluation score, and gamma 0.999 had a normal training reward (410.0 over the last 10 episodes) but the worst evaluation. Training episodes include random actions (epsilon 0.1 or more), while evaluation uses none. A lower loss did not mean a better agent either: gamma 0.90 had the lowest loss of the ten runs and ranked ninth.

### Combining changes

The chain baseline, `SILVER_EXP05`, `SILVER_EXP09`, `SILVER_EXP10` added one change at a time: batch size 256, then gamma 0.97, then learning rate 2e-4. The scores were 333.8, 516.2, 378.3 and 631.2, so the changes did not simply add up. Settings that are good individually have to be tested together.

### Final configuration

`SILVER_EXP10`: learning rate 0.0002, gamma 0.97, batch size 256, epsilon 1.0 to 0.1 over 100,000 steps.

It ranked first on the group's ten-episode protocol (592.5) and on the 60-episode evaluation (631.2), and it produced the highest single score (1,275). `SILVER_EXP05` is the runner-up and is more consistent from episode to episode.

## Limitations

- Each configuration was trained once with a single seed. The extended evaluation reduces the noise in measuring each trained model, but it cannot show how much the result would change with a different training seed.
- 100,000 steps is a short budget for Atari. The learning curves were still rising at the end, so the ranking could change with longer training.
- Experiments 1 to 8 change one hyperparameter at a time, so they do not show how the hyperparameters interact. Experiments 9 and 10 only test two combinations.

## Reproducibility

All files are under `experiments/silver/`:

- `configs/silver_experiments.json`: the ten configurations.
- `models/SILVER_EXPxx.pth`: the final weights of each experiment.
- `results/individual_experiments/SILVER_EXPxx/`: the exact configuration (`summary.json`), episode log (`episodes.csv`) and update log (`training_updates.csv`) of each experiment.
- `results/silver_*.csv`: training results, evaluation results and the random agent reference.

Any evaluation result can be reproduced with the group's `play.py`, which uses the same greedy protocol and seeds. For example, the 60-episode result of the final model:

```
uv run python play.py --model experiments/silver/models/SILVER_EXP10.pth --episodes 60 --no-render
```
