
from environment import make_env
import numpy as np


def main():
    env = make_env(render_mode="human")

    observation, info = env.reset(seed=42)

    # Check the initial observation.
    print("\n--- Q*bert Environment Verification ---")
    print("Environment started successfully!")
    print("Observation shape:", observation.shape)
    print("Observation data type:", observation.dtype)

    # Check the action space.
    action_count = env.action_space.n
    action_meanings = env.unwrapped.get_action_meanings()

    print("\n--- Available Actions ---")
    print("Number of actions:", action_count)

    for index, meaning in enumerate(action_meanings):
        print(f"Action {index}: {meaning}")

    # Verify the expected configuration.
    assert observation.shape == (4, 84, 84), "Incorrect observation shape"
    assert observation.dtype == np.uint8, "Expected uint8 observations"
    assert action_count == len(action_meanings), "Action count mismatch"
    assert action_count == 6, "Expected six Q*bert actions"

    print("\nObservation and action checks passed!")

    # Test that every available action can be executed.
    print("\n--- Testing Individual Actions ---")

    for action in range(action_count):
        observation, reward, terminated, truncated, info = env.step(action)

        print(
            f"Action {action} ({action_meanings[action]}): "
            f"Reward = {reward}"
        )

        assert observation.shape == (4, 84, 84)

        if terminated or truncated:
            observation, info = env.reset()

    # Test normal gameplay and episode endings.
    print("\n--- Testing Gameplay and Episode Resets ---")

    total_reward = 0
    completed_episodes = 0

    for step in range(2000):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            completed_episodes += 1
            print(
                f"Episode ended at step {step + 1}. "
                f"Reward accumulated so far: {total_reward}"
            )
            observation, info = env.reset()

    print("\n--- Verification Summary ---")
    print("Total test steps: 2000")
    print("Completed episodes:", completed_episodes)
    print("Total accumulated reward:", total_reward)
    print("Final observation shape:", observation.shape)

    env.close()

    print("\nEnvironment verification completed!")


if __name__ == "__main__":
    main()
