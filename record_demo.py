"""Record a video of a trained agent playing Q*bert (greedy actions, no randomness).

    uv run python record_demo.py --model models/ivan/TASK3_EXP04.pth --output demo/ivan_best_agent_EXP04.mp4

The agent plays several seeded games. The video shows the game with the highest score.
Only packages already in pyproject.toml are used (opencv for writing the video).
"""
import argparse
from pathlib import Path

import cv2
import torch

from dqn import DQN
from dqn_agent import get_device, select_action
from environment import make_env


def play_episode(env, model, device, seed):
    state, _ = env.reset(seed=seed)
    frames = [env.render()]
    total, done = 0.0, False
    while not done:
        action = select_action(model, state, 0.0, env.action_space.n, device)
        state, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        frames.append(env.render())
        done = terminated or truncated
    return total, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/ivan/TASK3_EXP04.pth"))
    parser.add_argument("--output", type=Path, default=Path("demo/ivan_best_agent_EXP04.mp4"))
    parser.add_argument("--episodes", type=int, default=10, help="games to try; the best one is recorded")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--gif", type=Path, default=Path("demo/ivan_best_agent_EXP04.gif"), help="also save a GIF for the README")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=int, default=3, help="enlarge the 160x210 game picture")
    args = parser.parse_args()

    device = get_device()
    env = make_env(render_mode="rgb_array")
    try:
        model = DQN(env.action_space.n).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
        model.eval()
        best_score, best_frames, best_seed = -1.0, None, None
        for i in range(args.episodes):
            score, frames = play_episode(env, model, device, args.seed + i)
            print(f"Game {i + 1} (seed {args.seed + i}): score {score:.0f}, {len(frames)} frames", flush=True)
            if score > best_score:
                best_score, best_frames, best_seed = score, frames, args.seed + i
    finally:
        env.close()

    height, width = best_frames[0].shape[:2]
    size = (width * args.scale, height * args.scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
    for frame in best_frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(cv2.resize(bgr, size, interpolation=cv2.INTER_NEAREST))
    writer.release()

    # A GIF plays directly inside the README on GitHub (the mp4 may need a download).
    if args.gif:
        from PIL import Image  # installed together with matplotlib

        step = 3  # keep every 3rd frame to keep the file small
        small = (width * 2, height * 2)
        images = [Image.fromarray(f).resize(small, Image.NEAREST) for f in best_frames[::step]]
        images[0].save(args.gif, save_all=True, append_images=images[1:], duration=int(1000 * step / args.fps), loop=0, optimize=True)
        print(f"Saved {args.gif}")
    print(f"Saved {args.output}: best game seed {best_seed}, score {best_score:.0f}, {len(best_frames) / args.fps:.1f} s")


if __name__ == "__main__":
    main()
