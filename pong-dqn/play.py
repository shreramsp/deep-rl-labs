#!/usr/bin/env python3
# ============================================================================
# Contributor: Sankalp Wahane
# THIS IS THE DEMO/PLAYBACK SCRIPT (not training).
# It loads a model file already trained by 1_baseline_dqn.py, then runs it with
# no randomness (pure exploitation) so you can watch/record how well it plays.
# ============================================================================
# FIXED: gym renamed/rebuilt as gymnasium.
import gymnasium as gym
import time
import argparse
import numpy as np

import torch

from lib import wrappers
from lib import dqn_model

import collections

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25  # playback speed cap so the visualization isn't sped-up/choppy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", help="Directory to store video recording")
    parser.add_argument("--no-visualize", default=True, action='store_false', dest='visualize',
                        help="Disable visualization of the game play")
    args = parser.parse_args()

    # FIXED for gymnasium: render_mode is now chosen when the env is created, not passed to
    # render() later. "rgb_array" is needed for recording, "human" pops up a live window.
    if args.record:
        render_mode = "rgb_array"
    elif args.visualize:
        render_mode = "human"
    else:
        render_mode = None
    env = wrappers.make_env(args.env, render_mode=render_mode)
    if args.record:
        # FIXED: gym.wrappers.Monitor was removed from gymnasium; RecordVideo replaces it.
        env = gym.wrappers.RecordVideo(env, args.record)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    state_dict = torch.load(args.model, map_location=lambda storage, loc: storage)
    # NoisyNet models (2a_noisynet_dqn.py) have the same layers plus noise parameters
    # (sigma_*, epsilon_*). Dropping those plays the learned mean weights with no noise —
    # the standard way to evaluate a NoisyNet greedily. Plain DQN models load unchanged.
    noise_keys = [k for k in state_dict if "sigma_" in k or "epsilon_" in k]
    if noise_keys:
        print("NoisyNet model: ignoring %d noise tensors, playing the noise-free mean network" % len(noise_keys))
        state_dict = {k: v for k, v in state_dict.items() if k not in noise_keys}
    net.load_state_dict(state_dict)

    # FIXED for gymnasium: reset() returns (observation, info) now.
    state, _ = env.reset()
    total_reward = 0.0
    c = collections.Counter()  # tallies how many times each action gets used, printed at the end

    while True:
        start_ts = time.time()
        if args.visualize:
            # FIXED: with render_mode="human" set at creation, step()/reset() already render
            # automatically each call — this explicit call is kept for parity with the book but
            # is effectively a harmless extra draw.
            env.render()
        # FIXED: same NumPy 2.x copy=False issue as in 1_baseline_dqn.py.
        state_v = torch.tensor(np.asarray([state]))
        q_vals = net(state_v).data.numpy()[0]
        # no epsilon-greedy here — always pick the action with the highest Q-value (pure exploitation,
        # since this is just showing off what the agent already learned, not training it further)
        action = np.argmax(q_vals)
        c[action] += 1
        # FIXED for gymnasium: step() returns 5 values now.
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if done:
            break
        if args.visualize:
            # cap playback speed to FPS so watching it isn't unrealistically fast
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    if args.record:
        # FIXED: env.close() (not env.env.close()) properly finalizes the RecordVideo wrapper too.
        env.close()

