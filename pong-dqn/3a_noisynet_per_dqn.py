#!/usr/bin/env python3
# ============================================================================
# Contributor: Sankalp Wahane
# STEP 3: NoisyNet DQN + Prioritized Experience Replay (PER)
# Identical to 2a_noisynet_dqn.py (the Step 2 winner) except for the replay buffer:
#   - 04: every stored transition is equally likely to be sampled for training (uniform)
#   - 07: transitions are sampled in proportion to how WRONG the network was on them
#         (their TD error), so training focuses on "surprising" experiences.
#
# PER paper: Schaul, Quan, Antonoglou & Silver, "Prioritized Experience Replay", ICLR 2016
# (proportional variant, alpha = 0.6, beta annealed from 0.4 up to 1.0).
# Settings as in Lapan, Deep RL Hands-On, Chapter 7 (05_dqn_prio_replay.py).
# Uses a sum tree so sampling and priority updates are O(log N) and vectorized in numpy:
# the book's version recomputed probabilities over the whole buffer every step and ran at
# ~71 f/s, which made it 32% SLOWER in wall-clock despite needing 37% fewer frames.
#
# NoisyNet paper: Fortunato et al., "Noisy Networks for Exploration", ICLR 2018.
# NoisyLinear follows the independent-Gaussian version the paper uses for DQN
# (sigma_0 = 0.017), as also implemented in Lapan, Deep RL Hands-On, Chapter 7.
# ============================================================================
from lib import wrappers

import argparse
import math
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.0  # same stop condition as the baseline

# --- identical to the baseline (1_baseline_dqn.py) so the comparison is fair ---
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

# NOTE: no EPSILON_* constants — NoisyNet replaces epsilon-greedy entirely.
SIGMA_INIT = 0.017  # starting noise scale from the NoisyNet paper (DQN variant)

# --- PER settings (Schaul et al., 2016; Lapan Ch. 7) ---
PRIO_ALPHA = 0.6        # how strongly priority shapes sampling (0 = uniform, 1 = fully by priority)
BETA_START = 0.4        # importance-sampling correction strength at the start...
BETA_FRAMES = 100000    # ...annealed linearly up to 1.0 (full correction) over this many frames
PRIO_EPS = 1e-5         # added to |TD error| so no transition ever gets zero chance


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class NoisyLinear(nn.Linear):
    """A normal Linear layer whose weights get random noise added on every forward pass:

        weight_used = weight + sigma_weight * random_normal_noise

    `weight` (mu) and `sigma_weight` are BOTH learned by backprop. So the network learns
    how much to "wiggle" each weight: where it is confident, gradient descent shrinks sigma
    (less exploration); where it is unsure, sigma stays large (keeps exploring)."""
    def __init__(self, in_features, out_features, sigma_init=SIGMA_INIT, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        # learnable noise scale, one per weight
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        # buffers hold the random noise sample (not learned, just re-drawn every forward)
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # mu initialization from the paper: uniform in [-sqrt(3/in), +sqrt(3/in)]
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        # draw fresh noise every call -> every action choice and every training step is "wiggled" differently
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)


class NoisyDQN(nn.Module):
    """Same architecture as lib/dqn_model.DQN; only the two fully connected layers are noisy."""
    def __init__(self, input_shape, n_actions):
        super(NoisyDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.noisy_layers = [
            NoisyLinear(conv_out_size, 512),
            NoisyLinear(512, n_actions)
        ]
        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

    def noisy_layers_sigma_snr(self):
        """Signal-to-noise ratio of each noisy layer: RMS(weight) / RMS(sigma).
        Rising SNR over training = the network is turning its own exploration down.
        Logged to TensorBoard — a good plot for the report."""
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]


class PrioritizedReplayBuffer:
    """Replay buffer where each transition i has a priority p_i and is sampled with
    probability P(i) = p_i^alpha / sum_k p_k^alpha.

    The p^alpha values live in a SUM TREE: a binary tree whose leaves are the priorities
    and whose every internal node holds the sum of its two children, so the root is the
    total. Sampling walks down from the root (go left if the random number fits in the
    left subtree's sum, else subtract it and go right): O(log N) per sample instead of
    O(N). All 64 samples of a batch walk the tree together as numpy vectors."""
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.data = [None] * capacity      # ring buffer of Experience tuples
        self.pos = 0                       # next slot to write (overwrites the oldest when full)
        self.size = 0
        self.tree_leaves = 1
        while self.tree_leaves < capacity:
            self.tree_leaves *= 2
        self.depth = int(math.log2(self.tree_leaves))
        self.tree = np.zeros(2 * self.tree_leaves, dtype=np.float64)  # node 1 = root, leaves at [L, 2L)
        self.max_priority = 1.0            # new transitions get the max seen so far -> sampled at least once soon

    def __len__(self):
        return self.size

    def _set(self, data_idx, values):
        """Write leaf values, then recompute every ancestor sum, one tree level at a time."""
        nodes = np.asarray(data_idx) + self.tree_leaves
        self.tree[nodes] = values
        nodes = np.unique(nodes // 2)
        while nodes[0] >= 1:
            self.tree[nodes] = self.tree[2 * nodes] + self.tree[2 * nodes + 1]
            if nodes[0] == 1:
                break
            nodes = np.unique(nodes // 2)

    def append(self, experience):
        self.data[self.pos] = experience
        self._set([self.pos], [self.max_priority ** self.alpha])
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta):
        total = self.tree[1]
        # stratified sampling: split [0, total) into batch_size equal segments, one draw per segment
        targets = (np.arange(batch_size) + np.random.random(batch_size)) * (total / batch_size)
        nodes = np.ones(batch_size, dtype=np.int64)
        for _ in range(self.depth):
            left = 2 * nodes
            left_sum = self.tree[left]
            go_left = targets < left_sum
            targets = np.where(go_left, targets, targets - left_sum)
            nodes = np.where(go_left, left, left + 1)
        indices = np.minimum(nodes - self.tree_leaves, self.size - 1)  # guard against float rounding at the edge

        # importance-sampling weights undo the bias of non-uniform sampling:
        # w_i = (N * P(i))^-beta, normalized by the max so weights only ever scale updates DOWN
        probs = self.tree[indices + self.tree_leaves] / total
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, dones, next_states = zip(*[self.data[idx] for idx in indices])
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.uint8), np.array(next_states)), indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        priorities = np.abs(td_errors) + PRIO_EPS
        self.max_priority = max(self.max_priority, float(priorities.max()))
        self._set(indices, priorities ** self.alpha)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state, _ = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, device="cpu", random_action=False):
        done_reward = None

        if random_action:
            # REPLAY WARM-UP ONLY (before any training): uniform random actions to fill the buffer,
            # as in the original DQN's "replay start size" (Mnih et al., 2015) and Dopamine's NoisyNet
            # agents. Without it, a freshly initialized NoisyNet picks almost the same action every
            # frame (noise ~0.03 vs ~0.15 gap between Q-values), so the buffer holds one action, the
            # other actions never get trained, and learning stalls. (Untuned run 1 sat at about -20.6
            # after 280k frames.) This matches the baseline, whose first 10k frames are also ~random (eps ~1).
            action = self.env.action_space.sample()
        else:
            # THE KEY DIFFERENCE FROM THE BASELINE: no coin flip, no random actions.
            # We always take the argmax — but the network's weights are noisy, so the
            # "best" action itself varies where the network is still uncertain.
            state_a = np.asarray([self.state])
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, weights, net, tgt_net, device="cpu"):
    """Same Bellman / DQN loss as the baseline, but each sample's squared error is multiplied
    by its importance-sampling weight. Also returns each sample's |TD error| (how wrong the
    network was), which becomes that transition's new priority."""
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    td_errors = state_action_values - expected_state_action_values
    weights_v = torch.tensor(weights).to(device)
    loss = (weights_v * td_errors ** 2).mean()
    return loss, td_errors.detach().abs().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    parser.add_argument("--no-random-warmup", default=False, action="store_true",
                        help="Disable uniform random actions while the replay buffer warms up "
                             "(reproduces untuned run 1, which stalled)")
    # --- TUNING knobs (the assignment's "and tune it"). Defaults = baseline behavior. ---
    #   --train-every N : do one gradient step every N frames instead of every frame
    #   --batch-size B  : samples per gradient step. Keeping B = 32 * N means the network still
    #                     learns from the same number of samples per frame played, in fewer,
    #                     larger steps (original DQN, Mnih et al. 2015, trained every 4 frames).
    parser.add_argument("--train-every", type=int, default=1,
                        help="Gradient step every N frames, default=1 (baseline)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Replay batch size, default=%d (baseline)" % BATCH_SIZE)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # tag goes into the TensorBoard run name and model filename, so different configs never mix
    run_tag = "-noisy-per-te%d-b%d%s" % (args.train_every, args.batch_size,
                                     "-nowarmup" if args.no_random_warmup else "-warmup")
    print("Config: random_warmup=%s, train_every=%d, batch_size=%d" % (
        not args.no_random_warmup, args.train_every, args.batch_size))

    env = wrappers.make_env(args.env)

    net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env + run_tag)
    print(net)

    buffer = PrioritizedReplayBuffer(REPLAY_SIZE, PRIO_ALPHA)
    agent = Agent(env, buffer)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    start_time = time.time()
    best_mean_reward = None

    while True:
        frame_idx += 1
        # importance-sampling correction grows from BETA_START to 1.0 (full correction)
        beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

        warming_up = len(buffer) < REPLAY_START_SIZE
        reward = agent.play_step(net, device=device,
                                 random_action=warming_up and not args.no_random_warmup)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            elapsed_hr = (time.time() - start_time) / 3600
            mean_reward = np.mean(total_rewards[-100:])
            snr = net.noisy_layers_sigma_snr()
            print("%d: done %d games, mean reward %.3f, snr %.2f/%.2f, speed %.2f f/s, elapsed %.3f hr" % (
                frame_idx, len(total_rewards), mean_reward, snr[0], snr[1], speed, elapsed_hr
            ), flush=True)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            writer.add_scalar("elapsed_hours", elapsed_hr, frame_idx)
            writer.add_scalar("beta", beta, frame_idx)
            for layer_idx, layer_snr in enumerate(snr, start=1):
                writer.add_scalar("snr_%d" % layer_idx, layer_snr, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + run_tag + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward), flush=True)
                best_mean_reward = mean_reward
            if mean_reward > args.reward:
                print("Solved in %d frames! (%.3f hr)" % (frame_idx, elapsed_hr), flush=True)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        if frame_idx % args.train_every != 0:
            # tuning: skip the gradient step on this frame (target sync above still counts frames)
            continue

        optimizer.zero_grad()
        batch, batch_indices, batch_weights = buffer.sample(args.batch_size, beta)
        loss_t, td_errors = calc_loss(batch, batch_weights, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
        buffer.update_priorities(batch_indices, td_errors)  # the transitions just trained on get fresh priorities
    writer.close()
