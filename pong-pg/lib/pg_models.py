# ============================================================================
# Networks for Project 2 (policy gradients on Pong).
# Both use the SAME convolutional body as the Project 1 DQN (book Chapter 6),
# so the only thing that changes between DQN, REINFORCE and A2C is the algorithm.
# Inputs are the Project 1 wrapper output: 4 stacked 84x84 frames scaled to 0-1.
# ============================================================================
import numpy as np
import torch
import torch.nn as nn


def conv_body(in_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
    )


def conv_out_size(body, shape):
    with torch.no_grad():
        return int(np.prod(body(torch.zeros(1, *shape)).size()))


class PolicyNet(nn.Module):
    """REINFORCE network: frames in, one logit per button out.
    softmax(logits) = the probability of pressing each of the 6 buttons."""
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = conv_body(input_shape[0])
        size = conv_out_size(self.conv, input_shape)
        self.policy = nn.Sequential(nn.Linear(size, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x):
        return self.policy(self.conv(x).view(x.size(0), -1))


class ActorCriticNet(nn.Module):
    """A2C network (book Chapter 10, AtariA2C): one shared body, two heads.
    actor  -> one logit per button (the policy)
    critic -> one number, V(s): the score the agent expects from this state on."""
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = conv_body(input_shape[0])
        size = conv_out_size(self.conv, input_shape)
        self.policy = nn.Sequential(nn.Linear(size, 512), nn.ReLU(), nn.Linear(512, n_actions))
        self.value = nn.Sequential(nn.Linear(size, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, x):
        out = self.conv(x).view(x.size(0), -1)
        return self.policy(out), self.value(out)
