
import numpy as np
import torch


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Stores transitions (state, action, reward, next_state, done) in a
    fixed-size circular buffer. When the buffer is full, the oldest
    experience is overwritten by the newest one.

    Why replay? Consecutive Atari frames are strongly correlated.
    Sampling random mini-batches from many past experiences breaks
    this correlation and lets each experience be reused many times.

    Memory saving:
        FrameStackObservation produces next_state by dropping the oldest
        frame of state and appending one new frame:

            next_state = [state[1], state[2], state[3], new_frame]

        So only the new frame (84 x 84) is stored for next_state, and the
        full next_state is rebuilt when sampling. This stores 5 frames per
        transition instead of 8 (about 35 KB instead of 56 KB).
    """

    def __init__(self, capacity, state_shape=(4, 84, 84), device="cpu"):
        self.capacity = capacity
        self.device = device

        # Pre-allocated storage. uint8 keeps images at 1 byte per pixel;
        # the DQN network converts them to floats in [0, 1] itself.
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_frames = np.zeros((capacity, *state_shape[1:]), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.position = 0  # Index where the next experience is written.
        self.size = 0      # Number of experiences currently stored.

    def add(self, state, action, reward, next_state, done):
        """
        Store one transition.

        done should be True only when the episode truly ended
        (terminated), not when it was cut off by a time limit
        (truncated). See train_step() in dqn_agent.py for why.
        """
        index = self.position

        self.states[index] = state
        self.next_frames[index] = next_state[-1]  # Newest frame only.
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Sample a random mini-batch of transitions as PyTorch tensors.

        Returns:
            states:      uint8   (batch_size, 4, 84, 84)
            actions:     int64   (batch_size,)
            rewards:     float32 (batch_size,)
            next_states: uint8   (batch_size, 4, 84, 84)
            dones:       float32 (batch_size,)
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        states = self.states[indices]

        # Rebuild next_state: last three frames of state + the new frame.
        next_states = np.concatenate(
            [states[:, 1:], self.next_frames[indices][:, None]],
            axis=1,
        )

        return (
            torch.as_tensor(states, device=self.device),
            torch.as_tensor(self.actions[indices], device=self.device),
            torch.as_tensor(self.rewards[indices], device=self.device),
            torch.as_tensor(next_states, device=self.device),
            torch.as_tensor(self.dones[indices], device=self.device),
        )

    def __len__(self):
        return self.size
