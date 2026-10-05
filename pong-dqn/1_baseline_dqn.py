#!/usr/bin/env python3
# ============================================================================
# Contributors: Sankalp Wahane (port of the book's Chapter 6 code to gymnasium / NumPy 2.x), Shreram Palanisamy (baseline training on Colab)
# THIS IS THE TRAINING SCRIPT.
# It builds a DQN agent (with replay buffer + target network) and trains it
# to play Atari Pong. Run this file to produce a trained model (a .dat file).
# ============================================================================
from lib import wrappers
from lib import dqn_model

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.0  # stop training once the 100-episode average score beats this ("solved")

GAMMA = 0.99                    # discount factor from the Bellman equation (how much future reward matters)
BATCH_SIZE = 32                 # how many stored experiences we sample from the replay buffer per training step
REPLAY_SIZE = 10000             # max size of the replay buffer (the "memory box" of past experiences)
LEARNING_RATE = 1e-4            # step size for the optimizer (Adam) when updating network weights
SYNC_TARGET_FRAMES = 1000       # how often (in frames) we copy net's weights into tgt_net (target network sync)
REPLAY_START_SIZE = 10000       # don't start training until the buffer has at least this many experiences

EPSILON_DECAY_LAST_FRAME = 10**5  # over how many frames epsilon shrinks from START to FINAL
EPSILON_START = 1.0                # epsilon-greedy: start fully random (100% exploration)
EPSILON_FINAL = 0.02               # epsilon-greedy: end mostly greedy (2% random, 98% exploit learned policy)


# One "row" stored in the replay buffer: what state we were in, what action we took,
# what reward we got, whether the episode ended, and what state we landed in next.
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    """The replay buffer from lecture: a fixed-size memory of past experiences.
    New experiences push out the oldest ones once it's full (deque = FIFO)."""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Pick random experiences from the buffer (this is the "shuffle" step from lecture
        # that breaks the correlation between consecutive game steps so the neural net can train properly).
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    """Wraps the environment + does the epsilon-greedy action selection from lecture."""
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        # FIXED for gymnasium: reset() returns (observation, info_dict) now.
        self.state, _ = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        # --- EPSILON-GREEDY (the algorithm from lecture) ---
        if np.random.random() < epsilon:
            # EXPLORE: with probability epsilon, take a completely random action
            action = env.action_space.sample()
        else:
            # EXPLOIT: otherwise, ask the neural net for its best-known action (highest Q-value)
            # FIXED: np.array(..., copy=False) raises an error under NumPy 2.x when a copy
            # can't be avoided. np.asarray() does the same "avoid copy if possible" job safely.
            state_a = np.asarray([self.state])
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # FIXED for gymnasium: step() returns 5 values now.
        # "terminated" = episode ended naturally (e.g. game over)
        # "truncated"  = episode was cut off artificially (e.g. time limit)
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated
        self.total_reward += reward

        # Store this step's experience in the replay buffer (this is what gets shuffled and
        # sampled later in calc_loss, instead of training on live, correlated game steps).
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    """This is the Bellman-equation / DQN loss from lecture:
    loss = ( Q(s,a) from the LIVE network  -  [reward + gamma * max_a' Q_hat(s',a') from the TARGET network] )^2
    net = the network being trained every step ("Q" in lecture notes)
    tgt_net = the frozen target network, only synced every SYNC_TARGET_FRAMES steps ("Q hat" in lecture notes)
    """
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    # FIXED (minor modernization): torch.ByteTensor is old/deprecated PyTorch style.
    # Modern PyTorch prefers an explicit bool tensor.
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)

    # Q(s,a): what the live network currently predicts for the action we actually took
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    # max_a' Q_hat(s',a'): the target network's best guess for the next state
    next_state_values = tgt_net(next_states_v).max(1)[0]
    # if the episode ended, there is no "next state" reward to add — zero it out
    next_state_values[done_mask] = 0.0
    # detach() = don't backpropagate through the target network, it's meant to stay frozen
    next_state_values = next_state_values.detach()

    # the Bellman target: immediate reward + discounted best future value (from target net)
    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # make_env() (in lib/wrappers.py) builds Pong + applies all the preprocessing wrappers
    # (frame skipping, resize to 84x84 grayscale, stack 4 frames, scale pixels to 0-1).
    env = wrappers.make_env(args.env)

    # "net" = the LIVE network, updated every training step (this is "Q" from lecture).
    # "tgt_net" = the TARGET network, only synced periodically (this is "Q hat" from lecture).
    # Both start with random, but different, weights.
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)  # this is what TensorBoard reads from later
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True:
        frame_idx += 1
        # epsilon-greedy decay: starts at 1.0 (all random) and linearly shrinks to 0.02
        # (mostly using the learned policy) over EPSILON_DECAY_LAST_FRAME frames.
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            # reward is only returned when an episode (one full game) just finished
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            # this rolling 100-episode average is the "mean score" the assignment keeps referring to
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon,
                speed
            ))
            # these 4 lines are what feed TensorBoard's graphs
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > args.reward:
                # "solved" once the 100-episode average beats MEAN_REWARD_BOUND (19.5) —
                # this is your baseline "mean score" and convergence time for the assignment.
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            # don't train until the replay buffer has enough experiences to sample a good random batch
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            # THE TARGET NETWORK SYNC from lecture: copy live net's weights into the frozen target net
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)          # random shuffled batch from the replay buffer
        loss_t = calc_loss(batch, net, tgt_net, device=device)  # Bellman/DQN loss
        loss_t.backward()                          # backpropagation
        optimizer.step()                           # update the live network's weights
    writer.close()
