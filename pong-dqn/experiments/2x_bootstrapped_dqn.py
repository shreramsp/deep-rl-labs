#!/usr/bin/env python3
# ============================================================================
# Contributor: Shreram Palanisamy
# STEP 2 — ALGORITHM B: Bootstrapped DQN
# Same DQN as 1_baseline_dqn.py (replay buffer + target network, same hyperparameters),
# with ONE change: epsilon-greedy is removed. Exploration instead comes from an
# ensemble of K Q-value "heads" sharing one conv body. At the start of every game
# one head is picked at random and followed greedily for the whole game.
#
# Paper: Osband et al., "Deep Exploration via Bootstrapped DQN", NeurIPS 2016.
# Settings taken from the paper's Atari experiments:
#   K = 10 heads, p = 1 (every head trains on every sample — the paper found this
#   performs about the same as real bootstrap masks and is cheaper), and the shared
#   conv gradient is normalized by 1/K.
# ============================================================================
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so lib/ is importable from experiments/
from lib import wrappers

import argparse
import math
import random
import time
import numpy as np
import collections

import torch
import torch.nn as nn
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

# NOTE: no EPSILON_* constants — the ensemble of heads replaces epsilon-greedy entirely.
N_HEADS = 10  # K from the paper


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class BootstrappedDQN(nn.Module):
    """One shared conv body (same as lib/dqn_model.DQN) + K independent fully connected heads.

    Each head is its own Q-function: Linear(conv_out, 512) -> ReLU -> Linear(512, n_actions).
    The K heads' weights are stored stacked in single tensors (shape [K, ...]) so all heads
    run in one batched matrix multiply instead of a Python loop of K small layers — same
    math, much less GPU overhead per training step."""
    def __init__(self, input_shape, n_actions, n_heads=N_HEADS):
        super(BootstrappedDQN, self).__init__()
        self.n_heads = n_heads

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        # head k's first layer is w1[k], b1[k]; second layer is w2[k], b2[k]
        self.w1 = nn.Parameter(torch.empty(n_heads, conv_out_size, 512))
        self.b1 = nn.Parameter(torch.empty(n_heads, 1, 512))
        self.w2 = nn.Parameter(torch.empty(n_heads, 512, n_actions))
        self.b2 = nn.Parameter(torch.empty(n_heads, 1, n_actions))
        # same default init as nn.Linear: uniform in [-1/sqrt(fan_in), +1/sqrt(fan_in)].
        # Every head gets DIFFERENT random weights — that difference is where the
        # ensemble's disagreement (and so the exploration) comes from.
        for w, b, fan_in in ((self.w1, self.b1, conv_out_size), (self.w2, self.b2, 512)):
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(w, -bound, bound)
            nn.init.uniform_(b, -bound, bound)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, head=None):
        """head=None -> Q-values from ALL heads, shape [K, batch, n_actions] (used for training).
        head=k     -> Q-values from head k only, shape [batch, n_actions] (used for acting)."""
        conv_out = self.conv(x).view(x.size()[0], -1)
        if head is not None:
            hidden = torch.relu(conv_out @ self.w1[head] + self.b1[head])
            return hidden @ self.w2[head] + self.b2[head]

        if conv_out.requires_grad:
            # All K heads backprop into the one shared conv body, which would make its
            # effective learning rate K times larger. The paper normalizes that gradient by 1/K.
            conv_out.register_hook(lambda grad: grad / self.n_heads)
        conv_k = conv_out.unsqueeze(0).expand(self.n_heads, -1, -1)     # [K, batch, conv_out]
        hidden = torch.relu(torch.baddbmm(self.b1, conv_k, self.w1))     # [K, batch, 512]
        return torch.baddbmm(self.b2, hidden, self.w2)                   # [K, batch, n_actions]


class ExperienceBuffer:
    """Same replay buffer as the baseline. (With p = 1 no per-head bootstrap masks are needed.)"""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer, n_heads):
        self.env = env
        self.exp_buffer = exp_buffer
        self.n_heads = n_heads
        self._reset()

    def _reset(self):
        self.state, _ = self.env.reset()
        self.total_reward = 0.0
        # THE KEY DIFFERENCE FROM THE BASELINE: pick one head at random for this whole game.
        # Each head is a different "opinion" of the game, so following one consistently for an
        # entire game explores a coherent strategy ("deep exploration"), instead of epsilon-greedy's
        # random twitches on individual frames.
        self.active_head = random.randrange(self.n_heads)

    def play_step(self, net, device="cpu"):
        done_reward = None

        # no coin flip, no random actions: act greedily with this game's head
        state_a = np.asarray([self.state])
        state_v = torch.tensor(state_a).to(device)
        q_vals_v = net(state_v, head=self.active_head)
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


def calc_loss(batch, net, tgt_net, device="cpu"):
    """The baseline's Bellman / DQN loss, computed for every head at once.
    Each head k is trained against ITS OWN target head k (as in the paper):
        loss_k = ( Q_k(s,a) - [r + gamma * max_a' Q_hat_k(s',a')] )^2
    and the total loss is the sum over heads."""
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)

    q_all = net(states_v)                                                   # [K, batch, n_actions]
    actions_k = actions_v.view(1, -1, 1).expand(q_all.size(0), -1, 1)       # same action for every head
    state_action_values = q_all.gather(2, actions_k).squeeze(-1)            # [K, batch]

    next_state_values = tgt_net(next_states_v).max(2)[0]                    # [K, batch]
    next_state_values[:, done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v   # rewards broadcast over heads
    per_head_loss = ((state_action_values - expected_state_action_values) ** 2).mean(dim=1)
    return per_head_loss.sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    parser.add_argument("--heads", type=int, default=N_HEADS,
                        help="Number of bootstrap heads K, default=%d" % N_HEADS)
    # --- TUNING knobs (the assignment's "and tune it"). Defaults = baseline behavior. ---
    # Untuned Bootstrapped DQN ran at ~60-70 f/s vs the baseline's ~110 f/s, because every
    # training step updates K big heads. These cut the per-frame training cost:
    #   --train-every N : do one gradient step every N frames instead of every frame
    #   --batch-size B  : samples per gradient step. Keeping B = 32 * N means the network
    #                     still learns from the same number of samples per frame played,
    #                     just in fewer, larger steps (original DQN, Mnih et al. 2015,
    #                     trained every 4 frames).
    parser.add_argument("--train-every", type=int, default=1,
                        help="Gradient step every N frames, default=1 (baseline)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Replay batch size, default=%d (baseline)" % BATCH_SIZE)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # tag goes into the TensorBoard run name and model filename, so tuned and untuned runs never mix
    run_tag = "-bootstrap-k%d-te%d-b%d" % (args.heads, args.train_every, args.batch_size)
    print("Config: heads=%d, train_every=%d, batch_size=%d" % (args.heads, args.train_every, args.batch_size))

    env = wrappers.make_env(args.env)

    net = BootstrappedDQN(env.observation_space.shape, env.action_space.n, args.heads).to(device)
    tgt_net = BootstrappedDQN(env.observation_space.shape, env.action_space.n, args.heads).to(device)
    writer = SummaryWriter(comment="-" + args.env + run_tag)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer, args.heads)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    start_time = time.time()
    best_mean_reward = None

    while True:
        frame_idx += 1

        reward = agent.play_step(net, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            elapsed_hr = (time.time() - start_time) / 3600
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, speed %.2f f/s, elapsed %.3f hr" % (
                frame_idx, len(total_rewards), mean_reward, speed, elapsed_hr
            ), flush=True)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            writer.add_scalar("elapsed_hours", elapsed_hr, frame_idx)
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
            # copies ALL heads, so each head k gets its own frozen target head k
            tgt_net.load_state_dict(net.state_dict())

        if frame_idx % args.train_every != 0:
            # tuning: skip the gradient step on this frame (target sync above still counts frames)
            continue

        optimizer.zero_grad()
        batch = buffer.sample(args.batch_size)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()
