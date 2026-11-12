"""Greedy evaluation of all ten Task 3 models with one identical protocol.

    uv run python evaluate_experiments.py --episodes 10

Every model plays the same seeded episodes (epsilon = 0). Output goes to
results/ivan/evaluation.json and is appended to IvanHyperparameter_experiment_results.md.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

from dqn import DQN
from dqn_agent import get_device, select_action
from environment import make_env
from run_experiments import TABLE, load_experiments

ROOT = Path(__file__).resolve().parent
MARK = "## Greedy evaluation"


def evaluate_model(path, episodes, base_seed, device):
    env = make_env()
    try:
        model = DQN(env.action_space.n).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        rewards = []
        for i in range(episodes):
            state, _ = env.reset(seed=base_seed + i)
            done, total = False, 0.0
            while not done:
                action = select_action(model, state, 0.0, env.action_space.n, device)
                state, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = terminated or truncated
            rewards.append(total)
        return rewards
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    device = get_device()
    specs = load_experiments()
    results = {}
    for spec in specs:
        exp_id = spec["experiment_id"]
        rewards = evaluate_model(ROOT / "models" / "ivan" / f"{exp_id}.pth", args.episodes, args.seed, device)
        results[exp_id] = {"rewards": rewards, "mean": statistics.mean(rewards), "std": statistics.pstdev(rewards)}
        print(f"{exp_id}: mean={results[exp_id]['mean']:.1f} std={results[exp_id]['std']:.1f}", flush=True)
    (ROOT / "results" / "ivan" / "evaluation.json").write_text(
        json.dumps({"episodes": args.episodes, "base_seed": args.seed, "results": results}, indent=2), encoding="utf-8")

    ranked = sorted(results.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    lines = [MARK, "",
             f"Each saved model played the same {args.episodes} seeded episodes (seeds {args.seed}-{args.seed + args.episodes - 1}) with greedy actions (epsilon = 0).",
             "", "| Rank | Experiment | Mean reward | Std | Min | Max |", "| --- | --- | --- | --- | --- | --- |"]
    for rank, (exp_id, r) in enumerate(ranked, 1):
        lines.append(f"| {rank} | {exp_id} | {r['mean']:.1f} | {r['std']:.1f} | {min(r['rewards']):.0f} | {max(r['rewards']):.0f} |")
    best, worst = ranked[0], ranked[-1]
    lines += ["", f"Highest greedy-evaluation reward: `{best[0]}` ({best[1]['mean']:.1f}). Lowest: `{worst[0]}` ({worst[1]['mean']:.1f}).", ""]

    text = TABLE.read_text(encoding="utf-8")
    text = text.split(MARK)[0].rstrip() + "\n\n" + "\n".join(lines)
    TABLE.write_text(text, encoding="utf-8")
    print(f"Best: {best[0]} ({best[1]['mean']:.1f})")


if __name__ == "__main__":
    main()
