
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Convolutional neural network for Atari DQN.

    Input:
        Four stacked grayscale frames of shape (4, 84, 84).

    Output:
        One Q-value for each available action.
    """

    def __init__(self, num_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=32,
            kernel_size=8,
            stride=4,
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
        )

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, state):

        # Convert image values from 0-255 to 0-1.
        state = state.float() / 255.0

        # Extract visual features.
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))

        # Flatten convolutional features.
        state = torch.flatten(state, start_dim=1)

        # Estimate Q-values.
        state = F.relu(self.fc1(state))
        q_values = self.fc2(state)

        return q_values
