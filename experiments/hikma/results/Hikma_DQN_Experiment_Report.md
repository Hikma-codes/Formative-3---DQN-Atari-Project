# DQN Atari Hyperparameter Experiment Report

## 1. Project Overview

This project investigates the effect of different hyperparameter
configurations on a Deep Q-Network (DQN) agent trained to play
the Atari Q*bert environment.

The experiments were conducted using PyTorch and Gymnasium.
The environment used was `ALE/Qbert-v5`.

Ten experiments were conducted, with each experiment using
100,000 environment training steps.

## 2. Experimental Setup

- Algorithm: Deep Q-Network (DQN)
- Environment: ALE/Qbert-v5
- Framework: PyTorch
- Hardware: NVIDIA Tesla T4 GPU
- Training steps per experiment: 100,000
- Random seed: 42
- Number of experiments: 10
- Evaluation episodes per model: 10
- Evaluation action selection: Greedy policy (epsilon = 0)

The same CNN architecture and environment preprocessing
were used across all ten experiments.

## 3. Hyperparameter Configurations

The experiments varied one hyperparameter at a time
relative to the baseline configuration.

| Experiment | Learning Rate | Gamma | Batch Size | Epsilon Start | Epsilon End | Epsilon Decay |
| --- | --- | --- | --- | --- | --- | --- |
| HIKMA_EXP02 | 0.000200 | 0.990 | 32 | 1.00 | 0.10 | 100000 |
| HIKMA_EXP09 | 0.000100 | 0.990 | 32 | 1.00 | 0.10 | 75000 |
| HIKMA_EXP10 | 0.000100 | 0.990 | 32 | 1.00 | 0.10 | 150000 |
| HIKMA_EXP06 | 0.000100 | 0.990 | 128 | 1.00 | 0.10 | 100000 |
| HIKMA_EXP01 | 0.000075 | 0.990 | 32 | 1.00 | 0.10 | 100000 |
| HIKMA_EXP05 | 0.000100 | 0.990 | 48 | 1.00 | 0.10 | 100000 |
| HIKMA_EXP04 | 0.000100 | 0.985 | 32 | 1.00 | 0.10 | 100000 |
| HIKMA_EXP08 | 0.000100 | 0.990 | 32 | 1.00 | 0.15 | 100000 |
| HIKMA_EXP07 | 0.000100 | 0.990 | 32 | 0.80 | 0.10 | 100000 |
| HIKMA_EXP03 | 0.000100 | 0.970 | 32 | 1.00 | 0.10 | 100000 |

## 4. Training Results

The training metric recorded for comparison was the mean
reward over the final ten completed training episodes.

These values provide an indication of the agent's
performance near the end of training. They are not
a substitute for independent evaluation.

| Experiment | Last 10 Mean Reward | Episodes | Optimizer Updates |
| --- | --- | --- | --- |
| HIKMA_EXP02 | 437.50 | 277 | 24501 |
| HIKMA_EXP09 | 417.50 | 271 | 24501 |
| HIKMA_EXP10 | 377.50 | 279 | 24501 |
| HIKMA_EXP06 | 342.50 | 266 | 24501 |
| HIKMA_EXP01 | 332.50 | 282 | 24501 |
| HIKMA_EXP05 | 332.50 | 279 | 24501 |
| HIKMA_EXP04 | 330.00 | 275 | 24501 |
| HIKMA_EXP08 | 312.50 | 288 | 24501 |
| HIKMA_EXP07 | 287.50 | 280 | 24501 |
| HIKMA_EXP03 | 282.50 | 278 | 24501 |

## 5. Evaluation Method

Each trained model was evaluated over 10 episodes.

The evaluation used:
- The same Q*bert environment configuration.
- The same episode seeds for every model.
- Greedy action selection, with epsilon set to zero.
- No additional model training or weight updates.

The evaluation mean reward was used to rank the models.

## 6. Evaluation Results

| Evaluation Rank | Experiment | Evaluation Mean Reward | Evaluation Std Reward | Minimum Reward | Maximum Reward |
| --- | --- | --- | --- | --- | --- |
| 1 | HIKMA_EXP02 | 585.00 | 148.23 | 350.00 | 750.00 |
| 2 | HIKMA_EXP03 | 525.00 | 175.20 | 225.00 | 675.00 |
| 3 | HIKMA_EXP05 | 502.50 | 167.68 | 300.00 | 675.00 |
| 4 | HIKMA_EXP04 | 437.50 | 319.56 | 225.00 | 900.00 |
| 5 | HIKMA_EXP08 | 365.00 | 12.91 | 350.00 | 375.00 |
| 6 | HIKMA_EXP09 | 342.50 | 89.01 | 225.00 | 450.00 |
| 7 | HIKMA_EXP06 | 317.50 | 44.17 | 250.00 | 375.00 |
| 8 | HIKMA_EXP07 | 270.00 | 25.82 | 250.00 | 300.00 |
| 9 | HIKMA_EXP01 | 265.00 | 215.45 | 0.00 | 525.00 |
| 10 | HIKMA_EXP10 | 192.50 | 58.98 | 125.00 | 250.00 |

## 7. Best Performing Configuration

The best model based on the evaluation mean reward was
**HIKMA_EXP02**.

Its hyperparameters were:

- Learning rate: 0.0002
- Gamma: 0.99
- Batch size: 32
- Epsilon start: 1.0
- Epsilon end: 0.1
- Epsilon decay: 100000

Its evaluation mean reward was
**585.00**.

Its evaluation standard deviation was
**148.23**.

## 8. Discussion

The results allow us to compare how changes in learning rate,
discount factor, batch size, and epsilon settings affected
the agent's training and evaluation performance.

The highest training reward and highest evaluation reward
should be considered separately. A model that performs well
during training does not necessarily achieve the highest
reward during independent evaluation.

The evaluation results provide the stronger basis for
selecting the final model.

## 9. Limitations

Each configuration was trained using one random seed.
Although all models received the same training budget,
the results may still be affected by randomness in the
environment and learning process.

A more extensive study could repeat the experiments
with multiple training seeds and compare the average
performance across those independent runs.

## 10. Conclusion

Ten DQN hyperparameter experiments were completed,
and all ten trained models were evaluated under the
same evaluation procedure.

The final model selection was based on evaluation
performance rather than training reward alone.