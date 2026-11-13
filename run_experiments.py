"""Run Task 3's ten independent experiments one at a time.

Usage:
    uv run python run_experiments.py --list
    uv run python run_experiments.py --experiment TASK3_EXP01 --max-steps 100000
    uv run python run_experiments.py --all --max-steps 100000
    uv run python run_experiments.py --export-best TASK3_EXP03
"""
import argparse
import json
import shutil
from pathlib import Path

from train import TrainConfig, run_training

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "ivan_experiments.json"
RESULTS = ROOT / "results" / "ivan"
TABLE = ROOT / "IvanHyperparameter_experiment_results.md"
OBSERVATIONS = ROOT / "ivan_observations.json"
EVAL_MARK = "## Greedy evaluation"


def load_experiments():
    specs = json.loads(EXPERIMENTS.read_text(encoding="utf-8"))
    ids = [s["experiment_id"] for s in specs]
    if len(specs) != 10 or len(set(ids)) != 10:
        raise ValueError("Expected ten uniquely identified experiments")
    combos = [tuple(s[k] for k in ("learning_rate", "gamma", "batch_size", "epsilon_start", "epsilon_end", "epsilon_decay")) for s in specs]
    if len(set(combos)) != 10:
        raise ValueError("Each experiment must have a different hyperparameter combination")
    return specs


def summary_for(exp_id):
    path = RESULTS / exp_id / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_table(specs):
    notes = json.loads(OBSERVATIONS.read_text(encoding="utf-8")) if OBSERVATIONS.exists() else {}
    old_text = TABLE.read_text(encoding="utf-8") if TABLE.exists() else ""
    headers = ["Experiment", "Learning rate", "Gamma", "Batch size", "Epsilon start", "Epsilon end", "Epsilon decay (steps)", "Observed behaviour"]
    lines = ["# Ivan — Task 3: Individual Hyperparameter Experiments", "", "Environment: `ALE/Qbert-v5`. All results must come from actual runs. No rewards are pre-filled.", "", "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for spec in specs:
        s = summary_for(spec["experiment_id"])
        if s and s["last_10_episode_mean_reward"] is not None:
            obs = (f"Completed {s['completed_episodes']} episodes in {s['total_steps']} steps; "
                   f"mean reward (last 10): {s['last_10_episode_mean_reward']:.2f}; "
                   f"mean episode length: {s['last_10_episode_mean_length']:.1f}; "
                   f"optimizer updates: {s['optimizer_updates']}. "
                   + notes.get("experiments", {}).get(spec["experiment_id"], "Add qualitative observations after reviewing gameplay."))
        elif s:
            obs = f"Run finished ({s['total_steps']} steps), but no episode completed; more training needed."
        else:
            obs = "Not run yet — record actual training and gameplay observations."
        cells = [str(spec.get(key)) for key in ("experiment_id", "learning_rate", "gamma", "batch_size", "epsilon_start", "epsilon_end", "epsilon_decay")]
        lines.append("| " + " | ".join(cells + [obs]) + " |")
    completed = [s for spec in specs if (s := summary_for(spec["experiment_id"])) and s["last_10_episode_mean_reward"] is not None]
    lines += ["", "## Interpretation", "", "Compare runs using the same total environment-step budget and the *last-10 completed episode mean raw reward* (a noisy training diagnostic). For a defensible best/worst claim, evaluate saved models greedily over multiple seeded episodes with a consistent protocol; training rewards alone are not sufficient.", ""]
    if completed:
        best = max(completed, key=lambda s: s["last_10_episode_mean_reward"])
        worst = min(completed, key=lambda s: s["last_10_episode_mean_reward"])
        lines += [f"Current highest training reward: `{best['experiment_id']}` ({best['last_10_episode_mean_reward']:.2f}).", f"Current lowest training reward: `{worst['experiment_id']}` ({worst['last_10_episode_mean_reward']:.2f}).", "", "These are training diagnostics, not controlled evaluation results.", ""]
    else:
        lines += ["Best and worst configurations: pending real experiment results and evaluation.", ""]
    if notes.get("interpretation"):
        lines += ["### Written interpretation", "", notes["interpretation"], ""]
    lines += ["## Reproducibility", "", "Every experiment saves its exact configuration to `results/ivan/TASK3_EXPxx/summary.json`, logs episodes to `episodes.csv`, optimization updates to `training_updates.csv`, plots to `plots/ivan/`, and final weights to `models/ivan/`.", ""]
    text = "\n".join(lines)
    if EVAL_MARK in old_text:  # keep the greedy evaluation section written by evaluate_experiments.py
        text = text.rstrip() + "\n\n" + old_text[old_text.index(EVAL_MARK):]
    TABLE.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--experiment", choices=[s["experiment_id"] for s in load_experiments()])
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--export-best", metavar="EXPERIMENT_ID", help="Copy selected trained weights to dqn_model.pth")
    parser.add_argument("--max-steps", type=int, help="Override step budget for test runs or full experiments")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    specs = load_experiments()
    if args.list:
        for spec in specs:
            print(spec)
    elif args.export_best:
        if args.export_best not in {s["experiment_id"] for s in specs}:
            parser.error("Unknown experiment ID")
        source = ROOT / "models" / "ivan" / f"{args.export_best}.pth"
        if not source.is_file():
            parser.error(f"Model does not exist: {source}. Run training first.")
        destination = ROOT / "dqn_model.pth"
        shutil.copy2(source, destination)
        print(f"Exported {source} -> {destination}; use with play.py")
    else:
        requested = specs if args.all else [next(s for s in specs if s["experiment_id"] == args.experiment)]
        for spec in requested:
            if summary_for(spec["experiment_id"]):
                print(f"Skipping {spec['experiment_id']}: existing results found.")
                continue
            config = TrainConfig(**spec, max_steps=args.max_steps if args.max_steps is not None else 100_000, device=args.device, output_root=str(ROOT))
            result = run_training(config)
            print(f"Completed {result['experiment_id']}: {result['completed_episodes']} episodes")
            write_table(specs)
    write_table(specs)
    print(f"Results table: {TABLE}")


if __name__ == "__main__":
    main()
