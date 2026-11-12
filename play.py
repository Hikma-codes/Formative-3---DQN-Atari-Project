"""Watch a trained Q*bert DQN model play using greedy actions.

    uv run python play.py --model dqn_model.pth
    uv run python play.py --model models/ivan/TASK3_EXP01.pth --episodes 3
"""
import argparse
from pathlib import Path

import torch

from dqn import DQN
from dqn_agent import get_device, select_action
from environment import make_env


def evaluate(model_path, episodes=3, render=True):
    device = get_device()
    env = make_env(render_mode="human" if render else None)
    try:
        model = DQN(env.action_space.n).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        rewards = []
        for episode in range(episodes):
            state, _ = env.reset(seed=42 + episode)
            finished = False
            total_reward = 0.0
            steps = 0
            while not finished:
                action = select_action(model, state, 0.0, env.action_space.n, device)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                steps += 1
                finished = terminated or truncated
            rewards.append(total_reward)
            print(f"Episode {episode + 1}: reward={total_reward:.1f}, steps={steps}")
        print(f"Mean reward across {episodes} episodes: {sum(rewards) / episodes:.1f}")
        return rewards
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Q*bert DQN agent")
    parser.add_argument("--model", type=Path, default=Path("dqn_model.pth"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--no-render", action="store_true", help="Evaluate without opening the game window")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}. Train or export a model first.")
    evaluate(args.model, args.episodes, not args.no_render)


if __name__ == "__main__":
    main()
