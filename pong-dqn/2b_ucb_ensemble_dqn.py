#!/usr/bin/env python3
# ============================================================================
# Contributor: Shreram Palanisamy
# STEP 2 — ALGORITHM B (replaces Bootstrapped DQN): UCB exploration with Q-ensembles
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
    run_tag = "-ucb-k%d-l%g-te%d-b%d%s" % (args.heads, args.ucb_lambda, args.train_every, args.batch_size,
                                          "-nowarmup" if args.no_random_warmup else "-warmup")
    print("Config: heads=%d, ucb_lambda=%g, random_warmup=%s, train_every=%d, batch_size=%d" % (
        args.heads, args.ucb_lambda, not args.no_random_warmup, args.train_every, args.batch_size))

    env = wrappers.make_env(args.env)

    net = EnsembleDQN(env.observation_space.shape, env.action_space.n, args.heads).to(device)
    tgt_net = EnsembleDQN(env.observation_space.shape, env.action_space.n, args.heads).to(device)
    writer = SummaryWriter(comment="-" + args.env + run_tag)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
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
