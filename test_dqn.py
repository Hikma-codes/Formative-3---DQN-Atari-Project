
import torch

from environment import make_env
from dqn import DQN


def main():
    env = make_env()

    observation, info = env.reset(seed=42)

    # Convert the observation to a PyTorch tensor.
    state = torch.as_tensor(
        observation,
        dtype=torch.float32
    ).unsqueeze(0)

    # Get the number of available actions.
    num_actions = env.action_space.n

    # Initialise the CNN.
    network = DQN(num_actions)

    # Test the forward pass.
    q_values = network(state)

    print("\n--- DQN Network Verification ---")
    print("Input shape:", state.shape)
    print("Output shape:", q_values.shape)
    print("Number of actions:", num_actions)
    print("Q-values:", q_values)

    # Verify the output.
    assert state.shape == (1, 4, 84, 84)
    assert q_values.shape == (1, num_actions)
    assert torch.isfinite(q_values).all(), "Q-values contain invalid numbers"

    # Test that gradients can flow through the network.
    loss = q_values.mean()
    loss.backward()

    gradients_exist = all(
        parameter.grad is not None
        for parameter in network.parameters()
    )

    assert gradients_exist, "Some network parameters have no gradients"

    print("\nForward pass successful!")
    print("Backward pass successful!")
    print("Network parameters receive gradients.")
    print("DQN verification completed!")

    env.close()


if __name__ == "__main__":
    main()
