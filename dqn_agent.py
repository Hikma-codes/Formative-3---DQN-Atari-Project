
import math
import random

import torch
import torch.nn.functional as F


def get_device():
    """
    Pick the fastest available device: NVIDIA GPU, Apple GPU, or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_epsilon(step, epsilon_start, epsilon_end, epsilon_decay, schedule="linear"):
    """
    Exploration rate for the current environment step.

    epsilon_decay is measured in environment steps for both schedules.

    linear:
        epsilon falls in a straight line from epsilon_start to
        epsilon_end over epsilon_decay steps, then stays at epsilon_end.
        This is the schedule used in the original DQN paper.

    exponential:
        epsilon = end + (start - end) * exp(-step / epsilon_decay)
        epsilon falls quickly at first, then slowly. After
        epsilon_decay steps it has covered about 63% of the distance
        from epsilon_start to epsilon_end.
    """
    if schedule == "linear":
        if step >= epsilon_decay:
            return epsilon_end
        fraction = step / epsilon_decay
        return epsilon_start + fraction * (epsilon_end - epsilon_start)

    if schedule == "exponential":
        return epsilon_end + (epsilon_start - epsilon_end) * math.exp(
            -step / epsilon_decay
        )

    raise ValueError(f"Unknown epsilon schedule: {schedule}")


def select_action(online_network, state, epsilon, num_actions, device):
    """
    Epsilon-greedy action selection.

    With probability epsilon: pick a random action (exploration).
    With probability 1 - epsilon: pick the action with the highest
    predicted Q-value (exploitation).

    For evaluation in play.py, call this with epsilon=0.0 to act
    fully greedily.
    """
    if random.random() < epsilon:
        return random.randrange(num_actions)

    # No gradients needed: we are only choosing an action, not learning.
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, device=device).unsqueeze(0)
        q_values = online_network(state_tensor)

    return int(q_values.argmax(dim=1).item())


def train_step(
    online_network,
    target_network,
    optimizer,
    replay_buffer,
    batch_size,
    gamma,
    clip_rewards=True,
    double_dqn=False,
    max_grad_norm=10.0,
):
    """
    One DQN update on a random mini-batch from the replay buffer.

    1. Predict Q(s, a) for the actions actually taken (online network).
    2. Build the Bellman target with the target network:
           target = r + gamma * max_a' Q_target(s', a') * (1 - done)
    3. Loss = Huber loss between prediction and target.
    4. Backpropagate and update the online network with the optimizer.

    Arguments:
        clip_rewards:  Clip rewards to [-1, 1] as in the DQN paper.
                       Q*bert gives rewards like 25 and 300, which would
                       make Q-values very large and training unstable.
        double_dqn:    Optional improvement (off by default). The online
                       network chooses the best next action and the target
                       network evaluates it, which reduces overestimation.
        max_grad_norm: Gradient clipping limit for stability. None = off.

    Returns:
        Dictionary with the loss and the average predicted Q-value,
        for logging by the training script.
    """
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    if clip_rewards:
        rewards = rewards.clamp(-1.0, 1.0)

    # 1. Q-values of all actions, then keep only the action taken.
    #    q_values: (batch, num_actions) -> current_q: (batch,)
    q_values = online_network(states)
    current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # 2. Bellman target. Computed without gradients: the target is a
    #    fixed label, so only the online network is trained.
    with torch.no_grad():
        if double_dqn:
            next_actions = online_network(next_states).argmax(dim=1, keepdim=True)
            next_q = target_network(next_states).gather(1, next_actions).squeeze(1)
        else:
            next_q = target_network(next_states).max(dim=1).values

        # If the episode ended, there is no future reward: (1 - done) = 0.
        target_q = rewards + gamma * next_q * (1.0 - dones)

    # 3. Huber loss: squared error for small errors, absolute error for
    #    large ones, so one surprising transition cannot cause a huge update.
    loss = F.smooth_l1_loss(current_q, target_q)

    # 4. Backpropagation and optimizer update.
    optimizer.zero_grad()
    loss.backward()

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(online_network.parameters(), max_grad_norm)

    optimizer.step()

    return {
        "loss": loss.item(),
        "mean_q": current_q.mean().item(),
    }


def update_target_network(online_network, target_network):
    """
    Copy the online network's weights into the target network.

    Between copies the target network stays frozen, so the Bellman
    target does not shift after every update. Without this, the
    network would be chasing a target that moves each time it learns.
    """
    target_network.load_state_dict(online_network.state_dict())
