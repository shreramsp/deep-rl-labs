#!/usr/bin/env python3
# ============================================================================
# Contributor: Shreram Palanisamy
# STEP 3 (second variant): UCB Q-ensemble + Prioritized Experience Replay (PER)
# Identical to 2b_ucb_ensemble_dqn.py (Step 2 algorithm B) except for the replay buffer,
# which is the same sum-tree proportional PER as 3a_noisynet_per_dqn.py
# (Schaul et al., ICLR 2016; alpha = 0.6, beta 0.4 -> 1.0 over 100k frames).
# Priority of a transition = its |TD error| averaged over the K heads (all heads train on
# every sampled transition, so the mean is the ensemble's overall "surprise").
#
# --- Description of the underlying 06 algorithm follows ---
# Same DQN as 1_baseline_dqn.py (replay buffer + target network, same hyperparameters),
# with ONE change: epsilon-greedy is removed. Exploration instead comes from an
# ensemble of K Q-value "heads" sharing one conv body, and on EVERY step the agent
# picks the action with the highest upper confidence bound across the heads:
#
#     a = argmax_a [ mean_k Q_k(s,a) + lambda * std_k Q_k(s,a) ]
#
# Paper: Chen, Sidor, Abbeel & Schulman, "UCB Exploration via Q-Ensembles", 2017
# (arXiv:1706.01502), Algorithm 2, lambda = 0.1. Their ensemble/UCB agents beat both
# Double DQN and Bootstrapped DQN, with UCB best on 30 of 49 Atari games.
#
# Why it replaces Bootstrapped DQN here: Bootstrapped DQN (experiments/2x_bootstrapped_dqn.py)
# plays each game with ONE randomly chosen head, and it plateaued at 18.1–18.6 because
# weaker heads keep dragging the 100-game mean down. UCB uses ALL heads on every move:
# agreement -> exploit the ensemble mean, disagreement (high std) -> try that action.
#
# Network, training loss, and per-head targets are identical to experiments/2x_bootstrapped_dqn.py
# (Osband et al., 2016: shared conv body, p = 1, conv gradient normalized by 1/K).
# ============================================================================
from lib import wrappers

import argparse
import math
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
N_HEADS = 10  # K from the paper (Chen et al. use K = 10)
UCB_LAMBDA = 0.1  # exploration bonus weight from the paper

# --- PER settings (Schaul et al., 2016; Lapan Ch. 7) ---
PRIO_ALPHA = 0.6        # how strongly priority shapes sampling (0 = uniform, 1 = fully by priority)
BETA_START = 0.4        # importance-sampling correction strength at the start...
BETA_FRAMES = 100000    # ...annealed linearly up to 1.0 (full correction) over this many frames
PRIO_EPS = 1e-5         # added to |TD error| so no transition ever gets zero chance



Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class EnsembleDQN(nn.Module):
    """One shared conv body (same as lib/dqn_model.DQN) + K independent fully connected heads.

    Each head is its own Q-function: Linear(conv_out, 512) -> ReLU -> Linear(512, n_actions).
    The K heads' weights are stored stacked in single tensors (shape [K, ...]) so all heads
    run in one batched matrix multiply instead of a Python loop of K small layers — same
    math, much less GPU overhead per training step."""
    def __init__(self, input_shape, n_actions, n_heads=N_HEADS):
        super(EnsembleDQN, self).__init__()
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
        """head=None -> Q-values from ALL heads, shape [K, batch, n_actions] (used for training AND acting).
        head=k     -> Q-values from head k only, shape [batch, n_actions] (not used by UCB)."""
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
    def __init__(self, env, exp_buffer, ucb_lambda):
        self.env = env
        self.exp_buffer = exp_buffer
        self.ucb_lambda = ucb_lambda
        self._reset()

    def _reset(self):
        self.state, _ = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, device="cpu", random_action=False):
        done_reward = None

        if random_action:
            # REPLAY WARM-UP ONLY (before any training): uniform random actions to fill the buffer.
            # This is the "replay start size" of the original DQN (Mnih et al., 2015), also listed in
            # Chen et al.'s hyperparameter table ("number of uniform random actions taken before
            # learning starts"). Same fix as NoisyNet: an untrained network picks nearly the same
            # action every frame, so without it the warm-up buffer would hold almost one action.
            action = self.env.action_space.sample()
        else:
            # THE KEY DIFFERENCE FROM THE BASELINE: no coin flip. Ask ALL heads, then pick the
            # action with the best "average opinion + bonus for disagreement" (upper confidence bound).
            state_a = np.asarray([self.state])
            state_v = torch.tensor(state_a).to(device)
            q_all = net(state_v)[:, 0, :]                    # [K, n_actions] for this one state
            ucb = q_all.mean(dim=0) + self.ucb_lambda * q_all.std(dim=0)
            action = int(torch.argmax(ucb).item())

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
    """The baseline's Bellman / DQN loss, computed for every head at once.
    Each head k is trained against ITS OWN target head k (as in the paper):
        loss_k = ( Q_k(s,a) - [r + gamma * max_a' Q_hat_k(s',a')] )^2
    and the total loss is the sum over heads.
    PER: each sample's squared error is multiplied by its importance-sampling weight, and the
    function also returns each sample's |TD error| averaged over heads (its new priority)."""
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
    td_errors = state_action_values - expected_state_action_values          # [K, batch]
    weights_v = torch.tensor(weights).to(device)                            # [batch], broadcast over heads
    per_head_loss = (weights_v * td_errors ** 2).mean(dim=1)
    return per_head_loss.sum(), td_errors.detach().abs().mean(dim=0).cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    parser.add_argument("--heads", type=int, default=N_HEADS,
                        help="Number of ensemble heads K, default=%d" % N_HEADS)
    parser.add_argument("--ucb-lambda", type=float, default=UCB_LAMBDA,
                        help="UCB exploration bonus weight, default=%.2f" % UCB_LAMBDA)
    parser.add_argument("--no-random-warmup", default=False, action="store_true",
                        help="Disable uniform random actions while the replay buffer warms up")
    # --- TUNING knobs (the assignment's "and tune it"). Defaults = baseline behavior. ---
    # Every training step updates K big heads (untuned Bootstrapped DQN with K = 10 ran at
    # ~60-70 f/s vs the baseline's ~110). These cut the per-frame training cost:
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
    run_tag = "-ucb-per-k%d-l%g-te%d-b%d%s" % (args.heads, args.ucb_lambda, args.train_every, args.batch_size,
                                          "-nowarmup" if args.no_random_warmup else "-warmup")
    print("Config: heads=%d, ucb_lambda=%g, random_warmup=%s, train_every=%d, batch_size=%d" % (
        args.heads, args.ucb_lambda, not args.no_random_warmup, args.train_every, args.batch_size))

    env = wrappers.make_env(args.env)

    net = EnsembleDQN(env.observation_space.shape, env.action_space.n, args.heads).to(device)
    tgt_net = EnsembleDQN(env.observation_space.shape, env.action_space.n, args.heads).to(device)
    writer = SummaryWriter(comment="-" + args.env + run_tag)
    print(net)

    buffer = PrioritizedReplayBuffer(REPLAY_SIZE, PRIO_ALPHA)
    agent = Agent(env, buffer, args.ucb_lambda)

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
            print("%d: done %d games, mean reward %.3f, speed %.2f f/s, elapsed %.3f hr" % (
                frame_idx, len(total_rewards), mean_reward, speed, elapsed_hr
            ), flush=True)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            writer.add_scalar("elapsed_hours", elapsed_hr, frame_idx)
            writer.add_scalar("beta", beta, frame_idx)
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
        batch, batch_indices, batch_weights = buffer.sample(args.batch_size, beta)
        loss_t, td_errors = calc_loss(batch, batch_weights, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
        buffer.update_priorities(batch_indices, td_errors)  # the transitions just trained on get fresh priorities
    writer.close()
