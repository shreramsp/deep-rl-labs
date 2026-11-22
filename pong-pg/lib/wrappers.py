# ============================================================================
# Contributor: Sankalp Wahane (gymnasium / ale-py port of the book's Chapter 6 wrappers)
# This file builds and wraps the raw Atari Pong environment so it's usable by
# a small neural network. Atari frames come out huge, in color, and one at a
# time — none of that is directly trainable. Each "wrapper" class below fixes
# one specific issue. They get chained together at the bottom in make_env().
# ============================================================================
import cv2
# FIXED: gym renamed/rebuilt as gymnasium. Aliasing it as "gym" means the rest of this
# file (gym.Wrapper, gym.ObservationWrapper, gym.spaces.Box, etc.) doesn't need renaming.
import gymnasium as gym
import gymnasium.spaces
# NEW REQUIREMENT (not in the original book code): Atari envs like Pong now live in the
# separate ale-py package. Just importing it registers "PongNoFrameskip-v4" etc. with
# gymnasium — without this import, gym.make() fails with "Environment doesn't exist".
import ale_py
import numpy as np
import collections


class FireResetEnv(gym.Wrapper):
    """Some Atari games (Pong included) sit idle until the agent presses FIRE
    to actually start the round. This wrapper auto-presses FIRE on every reset
    so the agent doesn't have to learn "press fire first" from scratch."""
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        # FIXED for gymnasium: reset() now returns (obs, info), step() now returns
        # (obs, reward, terminated, truncated, info). "done" = terminated or truncated.
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}


class MaxAndSkipEnv(gym.Wrapper):
    """Atari runs at a high frame rate with lots of near-duplicate frames. This wrapper
    repeats each chosen action for `skip` frames (default 4) and only returns one combined
    frame back to the agent — this massively speeds up training since the agent doesn't
    have to make a decision on every single raw frame. It also takes the pixel-wise max of
    the last 2 frames to fix Atari's flicker (some sprites only render on every other frame)."""
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for _ in range(self._skip):
            # FIXED for gymnasium: step() returns 5 values now.
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        # FIXED for gymnasium: reset() returns (obs, info) now.
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class ProcessFrame84(gym.ObservationWrapper):
    """Shrinks the raw Atari screen (210x160 color) down to an 84x84 grayscale image.
    A smaller, single-channel image means way fewer numbers for the CNN to crunch through
    on every single frame, without losing the information that actually matters (ball/paddle position)."""
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114  # RGB -> grayscale
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]  # crop out the scoreboard/borders, keep the play area
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    """PyTorch's Conv2d expects images as (channels, height, width), but gym/gymnasium
    gives images as (height, width, channels). This just reorders the axes to match."""
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Pixel values come in as 0-255 integers. Neural nets train much better on small,
    normalized numbers, so this rescales every pixel down to a 0.0-1.0 float."""
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    """A single frame can't show motion (which way is the ball moving?). This wrapper
    stacks the last n_steps (4) frames together into one observation, so the network can
    infer speed/direction from the sequence — this is why frames are stacked in lecture too."""
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self, **kwargs):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        # FIXED for gymnasium: reset() returns (obs, info) now, so unpack before
        # passing just the obs into self.observation(). Info is passed through unchanged.
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]  # shift older frames back
        self.buffer[-1] = observation       # newest frame goes in the last slot
        return self.buffer


def make_env(env_name, render_mode=None):
    """Chains every wrapper above onto the raw Atari env, in order:
    1. MaxAndSkipEnv  - repeat actions for 4 frames, dedupe flicker
    2. FireResetEnv   - auto-press FIRE to start each round
    3. ProcessFrame84 - shrink to 84x84 grayscale
    4. ImageToPyTorch - reorder axes for PyTorch's Conv2d
    5. BufferWrapper  - stack last 4 frames together (so motion is visible)
    6. ScaledFloatFrame - scale pixels to 0.0-1.0
    This is exactly what 1_baseline_dqn.py calls via wrappers.make_env(args.env).
    render_mode: leave as None for training (fastest, no rendering overhead).
    play.py passes "human" (to watch it play) or "rgb_array" (to record video)."""
    # FIXED for gymnasium: render_mode is now chosen at creation time, not passed to render() later.
    env = gym.make(env_name, render_mode=render_mode)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)
