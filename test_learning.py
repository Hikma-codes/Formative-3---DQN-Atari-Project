
import numpy as np
import torch

from environment import make_env
from dqn import DQN
from replay_buffer import ReplayBuffer
from dqn_agent import (
    get_epsilon,
    select_action,
    train_step,
    update_target_network,
)


def networks_equal(network_a, network_b):
    return all(
        torch.equal(a, b)
        for a, b in zip(network_a.parameters(), network_b.parameters())
    )


def test_replay_buffer(env):
    print("\n--- Replay Buffer ---")

    buffer = ReplayBuffer(capacity=500)
    state, info = env.reset(seed=42)
    real_next_states = []

    for step in range(300):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)

        buffer.add(state, action, reward, next_state, terminated)
        real_next_states.append(next_state.copy())

        state = next_state
        if terminated or truncated:
            state, info = env.reset()

    # The rebuilt next_state must match the real one exactly.
    for index in range(len(buffer)):
        rebuilt = np.concatenate(
            [buffer.states[index][1:], buffer.next_frames[index][None]]
        )
        assert np.array_equal(rebuilt, real_next_states[index]), (
            f"next_state rebuilt incorrectly at index {index}"
        )
    print("Rebuilt next_states match the environment exactly.")

    states, actions, rewards, next_states, dones = buffer.sample(32)

    assert states.shape == (32, 4, 84, 84) and states.dtype == torch.uint8
    assert next_states.shape == (32, 4, 84, 84) and next_states.dtype == torch.uint8
    assert actions.shape == (32,) and actions.dtype == torch.int64
    assert rewards.shape == (32,) and rewards.dtype == torch.float32
    assert dones.shape == (32,) and dones.dtype == torch.float32
    print("Sampled batch shapes and types are correct.")

    # Circular overwrite: a full buffer stays at capacity.
    small = ReplayBuffer(capacity=10)
    for _ in range(25):
        small.add(state, 0, 0.0, state, False)
    assert len(small) == 10 and small.position == 5
    print("Buffer overwrites oldest experiences when full.")


def test_epsilon():
    print("\n--- Epsilon Schedule ---")

    for schedule in ["linear", "exponential"]:
        values = [
            get_epsilon(step, 1.0, 0.05, 10_000, schedule)
            for step in [0, 5_000, 10_000, 100_000]
        ]
        print(f"{schedule:12s}", [round(v, 3) for v in values])

        assert values[0] == 1.0
        assert values[0] > values[1] > values[2] >= values[3] >= 0.05

    assert abs(get_epsilon(5_000, 1.0, 0.05, 10_000) - 0.525) < 1e-9
    assert get_epsilon(10_000, 1.0, 0.05, 10_000) == 0.05
    print("Epsilon decays from start to end correctly.")


def test_select_action(env, network, device):
    print("\n--- Epsilon-Greedy Action Selection ---")

    state, info = env.reset(seed=0)
    num_actions = env.action_space.n

    # epsilon = 0: always the greedy action.
    with torch.no_grad():
        q_values = network(torch.as_tensor(state, device=device).unsqueeze(0))
    greedy_action = int(q_values.argmax().item())

    greedy_choices = {
        select_action(network, state, 0.0, num_actions, device)
        for _ in range(50)
    }
    assert greedy_choices == {greedy_action}
    print("epsilon = 0 always selects the highest Q-value action.")

    # epsilon = 1: random actions covering the whole action space.
    random_choices = {
        select_action(network, state, 1.0, num_actions, device)
        for _ in range(500)
    }
    assert random_choices == set(range(num_actions))
    print("epsilon = 1 explores every action.")


def fill_buffer(env, buffer, steps, force_done=False):
    state, info = env.reset(seed=1)
    for _ in range(steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        buffer.add(state, action, reward, next_state, terminated or force_done)
        state = next_state
        if terminated or truncated:
            state, info = env.reset()


def test_train_step(env, device):
    print("\n--- Bellman Update, Loss and Target Network ---")

    num_actions = env.action_space.n
    online_network = DQN(num_actions).to(device)
    target_network = DQN(num_actions).to(device)
    update_target_network(online_network, target_network)
    assert networks_equal(online_network, target_network)
    print("Target network starts as an exact copy of the online network.")

    optimizer = torch.optim.Adam(online_network.parameters(), lr=1e-4)

    buffer = ReplayBuffer(capacity=200, device=device)
    fill_buffer(env, buffer, 200)

    target_before = [p.clone() for p in target_network.parameters()]
    online_before = [p.clone() for p in online_network.parameters()]

    result = train_step(
        online_network, target_network, optimizer, buffer,
        batch_size=32, gamma=0.99,
    )
    assert np.isfinite(result["loss"]) and np.isfinite(result["mean_q"])

    assert all(torch.equal(a, b) for a, b in
               zip(target_before, target_network.parameters()))
    assert not all(torch.equal(a, b) for a, b in
                   zip(online_before, online_network.parameters()))
    print("train_step updates the online network only; target stays frozen.")

    update_target_network(online_network, target_network)
    assert networks_equal(online_network, target_network)
    print("update_target_network copies the weights correctly.")

    # Learning check: every transition is marked done, so the target is
    # just the reward. Rewards are set to 1 for even actions and 0 for odd
    # ones, so the network must learn different Q-values per action.
    done_buffer = ReplayBuffer(capacity=64, device=device)
    fill_buffer(env, done_buffer, 64, force_done=True)
    done_buffer.rewards[:] = (done_buffer.actions % 2 == 0).astype(np.float32)

    losses = [
        train_step(online_network, target_network, optimizer, done_buffer,
                   batch_size=64, gamma=0.99)["loss"]
        for _ in range(150)
    ]
    print(f"Loss on a fixed batch: {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0] * 0.5, "Loss did not decrease"
    print("The network learns: loss decreases on a fixed batch.")


def test_mini_training_loop(env, device):
    print("\n--- Mini Training Loop (integration check) ---")

    num_actions = env.action_space.n
    online_network = DQN(num_actions).to(device)
    target_network = DQN(num_actions).to(device)
    update_target_network(online_network, target_network)
    optimizer = torch.optim.Adam(online_network.parameters(), lr=1e-4)
    buffer = ReplayBuffer(capacity=2_000, device=device)

    state, info = env.reset(seed=7)
    updates = 0

    for total_steps in range(1_500):
        epsilon = get_epsilon(total_steps, 1.0, 0.1, 1_000)
        action = select_action(online_network, state, epsilon, num_actions, device)

        next_state, reward, terminated, truncated, info = env.step(action)
        buffer.add(state, action, reward, next_state, terminated)
        state = next_state

        if terminated or truncated:
            state, info = env.reset()

        if total_steps >= 500 and total_steps % 4 == 0:
            result = train_step(online_network, target_network, optimizer,
                                buffer, batch_size=32, gamma=0.99)
            updates += 1
            if updates % 100 == 0:
                update_target_network(online_network, target_network)

    print(f"Ran 1500 steps, {updates} updates, final loss {result['loss']:.4f}, "
          f"final epsilon {epsilon:.2f}")
    print("All components work together.")


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")

    env = make_env()
    network = DQN(env.action_space.n).to(device)

    test_replay_buffer(env)
    test_epsilon()
    test_select_action(env, network, device)
    test_train_step(env, device)
    test_mini_training_loop(env, device)

    env.close()
    print("\nAll Member 2 checks passed!")


if __name__ == "__main__":
    main()
