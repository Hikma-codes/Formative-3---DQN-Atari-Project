"""Task 3: configurable Q*bert DQN training pipeline.

Uses the team's existing environment.py, dqn.py, replay_buffer.py and
the team's dqn_agent.py without changing any of their interfaces.

Run with: uv run python train.py --experiment-id TASK3_EXP01 --max-steps 100000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from dqn import DQN
from dqn_agent import (
    get_device,
    get_epsilon,
    select_action,
    train_step,
    update_target_network,
)
from environment import ENV_NAME, make_env
from replay_buffer import ReplayBuffer


@dataclass
class TrainConfig:
    experiment_id: str = "TASK3_EXP01"
    member: str = "ivan"
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 32
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: int = 100_000  # environment decisions/steps, not episodes
    epsilon_schedule: str = "linear"
    max_steps: int = 100_000
    replay_capacity: int = 20_000
    learning_starts: int = 2_000
    train_frequency: int = 4
    target_update_frequency: int = 2_000  # environment steps
    checkpoint_frequency: int = 10_000  # environment steps
    log_frequency: int = 1_000
    seed: int = 42
    output_root: str = "."
    device: str = "auto"
    clip_rewards: bool = True
    double_dqn: bool = False
    max_grad_norm: float = 10.0

    def validate(self) -> None:
        if not self.experiment_id or "/" in self.experiment_id or "\\" in self.experiment_id or self.experiment_id in {".", ".."}:
            raise ValueError("experiment_id must be a simple nonempty filename")
        if not self.member or "/" in self.member or "\\" in self.member or self.member in {".", ".."}:
            raise ValueError("member must be a simple nonempty folder name")
        if self.learning_rate <= 0 or not (0 <= self.gamma <= 1):
            raise ValueError("learning_rate must be positive; gamma must be in [0, 1]")
        if not (0 <= self.epsilon_end <= self.epsilon_start <= 1):
            raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")
        if self.epsilon_schedule not in {"linear", "exponential"}:
            raise ValueError("epsilon_schedule must be linear or exponential")
        for field in (
            "batch_size", "epsilon_decay", "max_steps", "replay_capacity",
            "learning_starts", "train_frequency", "target_update_frequency",
            "checkpoint_frequency", "log_frequency",
        ):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size cannot exceed replay_capacity")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


def save_model(network: DQN, path: Path) -> None:
    """Save *weights only* so existing play.py can load_state_dict(torch.load(...))."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), path)


def _write_plot(rows: list[dict], plot_path: Path) -> None:
    # Noninteractive backend works on servers and CI.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    steps = [r["total_steps"] for r in rows]
    rewards = [r["reward"] for r in rows]
    lengths = [r["episode_length"] for r in rows]
    axes[0].plot(steps, rewards, linewidth=1, label="Episode reward")
    if rewards:
        window = min(20, len(rewards))
        moving = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axes[0].plot(steps[window - 1:], moving, label=f"Moving mean ({window})")
    axes[0].set_ylabel("Raw episode reward")
    axes[0].legend()
    axes[1].plot(steps, lengths)
    axes[1].set_ylabel("Episode length (decisions)")
    axes[2].plot(steps, [r["mean_loss"] for r in rows])
    axes[2].set_ylabel("Mean loss since last episode")
    axes[2].set_xlabel("Environment steps")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Q*bert DQN training: episode metrics")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def run_training(config: TrainConfig) -> dict:
    """Train one independent run and return metrics (never invent results)."""
    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = get_device() if config.device == "auto" else torch.device(config.device)
    root = Path(config.output_root)
    model_dir = root / "models" / config.member
    results_dir = root / "results" / config.member / config.experiment_id
    plot_dir = root / "plots" / config.member
    for path in (model_dir, results_dir, plot_dir):
        path.mkdir(parents=True, exist_ok=True)

    env = make_env()
    started = time.monotonic()
    try:
        num_actions = env.action_space.n
        online_network = DQN(num_actions).to(device)
        target_network = DQN(num_actions).to(device)
        update_target_network(online_network, target_network)
        target_network.eval()
        optimizer = torch.optim.Adam(online_network.parameters(), lr=config.learning_rate)
        replay_buffer = ReplayBuffer(capacity=config.replay_capacity, device=device)

        state, _ = env.reset(seed=config.seed)
        if tuple(state.shape) != (4, 84, 84):
            raise ValueError(f"Unexpected observation shape: {state.shape}")
        episode_reward = 0.0
        episode_length = 0
        episode = 0
        updates = 0
        losses: list[float] = []
        episode_rows: list[dict] = []
        log_path = results_dir / "episodes.csv"
        loss_path = results_dir / "training_updates.csv"
        log_fields = ["episode", "total_steps", "episode_length", "reward", "epsilon", "mean_loss", "updates"]
        loss_fields = ["total_steps", "update", "loss", "mean_q"]
        with log_path.open("w", newline="", encoding="utf-8") as episode_file, loss_path.open("w", newline="", encoding="utf-8") as update_file:
            episode_writer = csv.DictWriter(episode_file, fieldnames=log_fields)
            loss_writer = csv.DictWriter(update_file, fieldnames=loss_fields)
            episode_writer.writeheader()
            loss_writer.writeheader()

            for total_steps in range(1, config.max_steps + 1):
                epsilon = get_epsilon(
                    total_steps - 1, config.epsilon_start, config.epsilon_end,
                    config.epsilon_decay, schedule=config.epsilon_schedule,
                )
                action = select_action(online_network, state, epsilon, num_actions, device)
                next_state, reward, terminated, truncated, _ = env.step(action)
                # Bellman bootstrap stops on terminated; truncation is a reset,
                # but should NOT be represented as terminal in the buffer.
                replay_buffer.add(state, action, reward, next_state, terminated)
                episode_reward += float(reward)
                episode_length += 1
                state = next_state

                if (
                    total_steps >= config.learning_starts
                    and len(replay_buffer) >= config.batch_size
                    and total_steps % config.train_frequency == 0
                ):
                    training_result = train_step(
                        online_network, target_network, optimizer, replay_buffer,
                        config.batch_size, config.gamma, clip_rewards=config.clip_rewards,
                        double_dqn=config.double_dqn, max_grad_norm=config.max_grad_norm,
                    )
                    updates += 1
                    losses.append(training_result["loss"])
                    loss_writer.writerow({
                        "total_steps": total_steps, "update": updates,
                        "loss": training_result["loss"],
                        "mean_q": training_result["mean_q"],
                    })

                if total_steps % config.target_update_frequency == 0:
                    update_target_network(online_network, target_network)

                if terminated or truncated:
                    episode += 1
                    entry = {
                        "episode": episode,
                        "total_steps": total_steps,
                        "episode_length": episode_length,
                        "reward": episode_reward,
                        "epsilon": epsilon,
                        "mean_loss": float(np.mean(losses)) if losses else 0.0,
                        "updates": updates,
                    }
                    episode_rows.append(entry)
                    episode_writer.writerow(entry)
                    episode_file.flush()
                    update_file.flush()
                    episode_reward = 0.0
                    episode_length = 0
                    losses.clear()
                    state, _ = env.reset()

                if total_steps % config.checkpoint_frequency == 0:
                    save_model(online_network, results_dir / f"checkpoint_step_{total_steps}.pth")
                if total_steps % config.log_frequency == 0:
                    last_reward = episode_rows[-1]["reward"] if episode_rows else None
                    print(f"[{config.experiment_id}] steps={total_steps}/{config.max_steps} "
                          f"episodes={episode} updates={updates} eps={epsilon:.3f} "
                          f"last_reward={last_reward}", flush=True)

        final_path = model_dir / f"{config.experiment_id}.pth"
        save_model(online_network, final_path)
        # Standard filename for the final selected best experiment is exported
        # separately, to avoid overwriting it with each of ten runs.
        plot_path = plot_dir / f"{config.experiment_id}.png"
        _write_plot(episode_rows, plot_path)
        recent = episode_rows[-min(10, len(episode_rows)):]
        summary = {
            "experiment_id": config.experiment_id,
            "member": config.member,
            "environment": ENV_NAME,
            "config": asdict(config),
            "device": str(device),
            "total_steps": config.max_steps,
            "completed_episodes": episode,
            "optimizer_updates": updates,
            "last_10_episode_mean_reward": float(np.mean([r["reward"] for r in recent])) if recent else None,
            "last_10_episode_mean_length": float(np.mean([r["episode_length"] for r in recent])) if recent else None,
            "best_episode_reward": max((r["reward"] for r in episode_rows), default=None),
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "model_path": str(final_path),
            "episode_log": str(log_path),
            "update_log": str(loss_path),
            "plot_path": str(plot_path),
            "partial_episode_discarded_from_summary": episode_length > 0,
        }
        (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Q*bert DQN using existing group modules")
    parser.add_argument("--config", type=Path, help="JSON object of TrainConfig fields")
    for name, field in TrainConfig.__dataclass_fields__.items():
        default = field.default
        cli_name = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(cli_name, dest=name, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(cli_name, dest=name, type=type(default), default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_data = {}
    if args.config:
        config_data = json.loads(args.config.read_text(encoding="utf-8"))
    overrides = {field: getattr(args, field) for field in TrainConfig.__dataclass_fields__ if getattr(args, field) is not None}
    config = TrainConfig(**{**config_data, **overrides})
    summary = run_training(config)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
