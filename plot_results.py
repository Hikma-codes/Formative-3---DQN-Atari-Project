"""
plot_results.py - Compare all hyperparameter experiments of one member.

train.py already saves a training plot for each single experiment in
plots/<member>/. This script puts all of a member's experiments side by side.

It reads the files train.py writes for every experiment:
    results/<member>/<experiment_id>/episodes.csv
    results/<member>/<experiment_id>/summary.json

Usage:
    uv run python plot_results.py --member yvette
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BLUE = "#2a78d6"
INK = "#333333"
SMOOTHING_WINDOW = 20  # same moving-average window as train.py's plots


def load_experiments(member):
    """Return a list of (experiment_id, episode_log, summary) for one member."""
    experiments = []
    for results_dir in sorted(Path("results", member).iterdir()):
        log_path = results_dir / "episodes.csv"
        summary_path = results_dir / "summary.json"
        if log_path.exists() and summary_path.exists():
            episode_log = pd.read_csv(log_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            experiments.append((results_dir.name, episode_log, summary))
    return experiments


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

    for ax, (experiment_id, episode_log, _) in zip(axes.flat, experiments):
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
    experiment_ids = [experiment_id for experiment_id, _, _ in experiments]
    final_rewards = [summary["last_10_episode_mean_reward"] or 0 for _, _, summary in experiments]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.bar(experiment_ids, final_rewards, color=BLUE)
    ax.bar_label(bars, fmt="%.0f", color=INK, fontsize=9)
    style_axis(ax, f"{member}: mean reward over the last 10 training episodes")
    ax.set_ylabel("Reward", color=INK)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    save_figure(fig, output_dir / "final_reward_comparison.png")


def save_figure(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare one member's DQN experiments.")
    parser.add_argument("--member", required=True, help="member folder name, e.g. yvette")
    args = parser.parse_args()

    experiments = load_experiments(args.member)
    if not experiments:
        print(f"No finished experiments found in results/{args.member}/")
        return

    output_dir = Path("plots", args.member)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_reward_curves(args.member, experiments, output_dir)
    plot_final_rewards(args.member, experiments, output_dir)


if __name__ == "__main__":
    main()