"""
plot_results.py - Training plots for all hyperparameter experiments of one member.

It reads the episode log train.py writes for every experiment:
    results/<member>/<experiment_id>/episodes.csv

and creates, in plots/<member>/:
    <experiment_id>.png            reward, episode length and loss for one experiment
                                   (drawn with train.py's own plotting function)
    reward_curves_comparison.png   all experiments side by side
    final_reward_comparison.png    mean reward of the last 10 episodes per experiment

Usage:
    uv run python plot_results.py --member yvette
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from train import _write_plot

BLUE = "#2a78d6"
INK = "#333333"
SMOOTHING_WINDOW = 20  # same moving-average window as train.py's plots
LAST_EPISODES = 10     # same "last 10 episodes" measure as train.py's summary


def load_experiments(member):
    """Return a list of (experiment_id, episode_log) for one member."""
    experiments = []
    for results_dir in sorted(Path("results", member).iterdir()):
        log_path = results_dir / "episodes.csv"
        if log_path.exists():
            experiments.append((results_dir.name, pd.read_csv(log_path)))
    return experiments


def print_results_table(experiments):
    """Print the numbers needed for the hyperparameter experiment table."""
    print(f"\n{'Experiment':<16}{'Episodes':>10}{f'Last {LAST_EPISODES} mean':>16}{'Best episode':>14}")
    for experiment_id, episode_log in experiments:
        last_mean = episode_log["reward"].tail(LAST_EPISODES).mean()
        print(f"{experiment_id:<16}{len(episode_log):>10}{last_mean:>16.1f}{episode_log['reward'].max():>14.0f}")
    print()


def style_axis(ax, title):
    ax.set_title(title, color=INK, loc="left", fontsize=10)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def plot_reward_curves(member, experiments, output_dir):
    """One small panel per experiment, all on the same axes so they compare fairly."""
    columns = 5
    rows = math.ceil(len(experiments) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(18, 3.5 * rows),
                             sharex=True, sharey=True, squeeze=False)

    for ax, (experiment_id, episode_log) in zip(axes.flat, experiments):
        smoothed_reward = episode_log["reward"].rolling(SMOOTHING_WINDOW, min_periods=1).mean()
        ax.plot(episode_log["total_steps"], smoothed_reward, color=BLUE, linewidth=2)
        style_axis(ax, experiment_id)

    # Hide empty panels when a member has fewer than 10 experiments
    for ax in list(axes.flat)[len(experiments):]:
        ax.set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("Environment steps", color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("Episode reward", color=INK)

    fig.suptitle(f"{member}: episode reward per experiment ({SMOOTHING_WINDOW}-episode moving average)",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save_figure(fig, output_dir / "reward_curves_comparison.png")


def plot_final_rewards(member, experiments, output_dir):
    """Bar chart of each experiment's mean reward over its last 10 episodes."""
    experiment_ids = [experiment_id for experiment_id, _ in experiments]
    final_rewards = [episode_log["reward"].tail(LAST_EPISODES).mean() for _, episode_log in experiments]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.bar(experiment_ids, final_rewards, color=BLUE)
    ax.bar_label(bars, fmt="%.0f", color=INK, fontsize=9)
    style_axis(ax, f"{member}: mean reward over the last {LAST_EPISODES} training episodes")
    ax.set_ylabel("Reward", color=INK)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save_figure(fig, output_dir / "final_reward_comparison.png")


def save_figure(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot one member's DQN experiments.")
    parser.add_argument("--member", required=True, help="member folder name, e.g. yvette")
    args = parser.parse_args()

    experiments = load_experiments(args.member)
    if not experiments:
        print(f"No experiment logs found in results/{args.member}/")
        return

    output_dir = Path("plots", args.member)
    output_dir.mkdir(parents=True, exist_ok=True)

    # One training plot per experiment, using the team's plotting function from train.py
    for experiment_id, episode_log in experiments:
        plot_path = output_dir / f"{experiment_id}.png"
        _write_plot(episode_log.to_dict("records"), plot_path)
        print(f"Saved {plot_path}")

    plot_reward_curves(args.member, experiments, output_dir)
    plot_final_rewards(args.member, experiments, output_dir)
    print_results_table(experiments)


if __name__ == "__main__":
    main()