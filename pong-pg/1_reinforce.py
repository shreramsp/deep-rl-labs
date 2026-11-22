#!/usr/bin/env python3
# ============================================================================
# Project 2, baseline: REINFORCE (Williams, 1992) on Atari Pong.
#
# Same settings as Project 1: PongNoFrameskip-v4, the Project 1 wrappers
# (lib/wrappers.py), the same conv network body, gamma 0.99, Adam with
# learning rate 1e-4, and "solved" = 100-game mean reward above 19.0.
# The training loop follows the book's Chapter 9 REINFORCE
# (02_cartpole_reinforce.py: play whole games, compute discounted returns,
# loss = -mean(return * log pi(action | state)), one update, throw the games away).
#
# Plain English: play a few games, then make every move that came before
# winning a point more likely, and every move before losing a point less likely.
#
# Defaults = plain REINFORCE. Tuning flags (all off by default):
#   --lr, --games-per-update, --entropy-beta, --return-norm, --grad-clip,
#   --point-returns (a move's return stops at the next point scored, as in
#   Karpathy's "Pong from Pixels", 2016: each point is judged on its own)
# ============================================================================
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim
from tensorboardX import SummaryWriter

from lib import wrappers
from lib.pg_models import PolicyNet

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.0
GAMMA = 0.99
CHUNK = 1024  # states per forward/backward pass; only limits GPU memory, the math is unchanged


def discounted_returns(rewards, gamma, reset_on_point=False):
    """G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ... to the end of the game (book's calc_qvals).
    reset_on_point: in Pong a reward is non-zero only when a point ends, so restarting the sum there
    makes each move's return depend only on the point it belongs to (Karpathy, 2016)."""
    out, running = [], 0.0
    for r in reversed(rewards):
        if reset_on_point and r != 0:
            running = 0.0
        running = r + gamma * running
        out.append(running)
    return out[::-1]


def to_uint8(state):
    # The wrapper gives floats 0-1 that came from 0-255 pixels; storing them as bytes is exact and 4x smaller.
    return np.rint(state * 255.0).astype(np.uint8)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", action="store_true", help="Train on the GPU")
    p.add_argument("--env", default=DEFAULT_ENV_NAME)
    p.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND, help="Stop when the 100-game mean exceeds this")
    p.add_argument("--max-hours", type=float, default=0.0, help="Stop after this many hours (0 = no limit)")
    p.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate (Project 1 used 1e-4)")
    p.add_argument("--games-per-update", type=int, default=4, help="Games played before each update (book: 4)")
    p.add_argument("--entropy-beta", type=float, default=0.0, help="Entropy bonus weight (0 = plain REINFORCE)")
    p.add_argument("--return-norm", choices=["none", "mean", "meanstd"], default="none",
                   help="Subtract the batch mean return (and divide by its std) before the update")
    p.add_argument("--grad-clip", type=float, default=0.0, help="Clip the gradient norm (0 = off)")
    p.add_argument("--point-returns", action="store_true",
                   help="End each move's return at the next point scored (Pong-specific, Karpathy 2016)")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    tag = "reinforce-lr%g-g%d" % (args.lr, args.games_per_update)
    if args.entropy_beta > 0:
        tag += "-ent%g" % args.entropy_beta
    if args.return_norm != "none":
        tag += "-norm" + args.return_norm
    if args.grad_clip > 0:
        tag += "-clip%g" % args.grad_clip
    if args.point_returns:
        tag += "-pt"
    print("Config:", tag, vars(args), flush=True)

    env = wrappers.make_env(args.env)
    if args.seed is not None:
        env.reset(seed=args.seed)
    net = PolicyNet(env.observation_space.shape, env.action_space.n).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    writer = SummaryWriter(comment="-%s-%s" % (args.env, tag))
    print(net)

    total_rewards, best_mean = [], None
    frame_idx, update_idx = 0, 0
    start = ts = time.time()
    ts_frame = 0
    batch_states, batch_actions, batch_returns = [], [], []
    games_in_batch = 0
    state, _ = env.reset()
    ep_rewards, ep_reward_sum = [], 0.0
    stop_reason = None

    while stop_reason is None:
        # ---- play one step: sample a button from the policy's probabilities ----
        with torch.no_grad():
            logits = net(torch.from_numpy(np.asarray(state)).unsqueeze(0).to(device))
            action = int(torch.distributions.Categorical(logits=logits).sample().item())
        new_state, reward, terminated, truncated, _ = env.step(action)
        frame_idx += 1
        batch_states.append(to_uint8(state))
        batch_actions.append(action)
        ep_rewards.append(reward)
        ep_reward_sum += reward
        state = new_state
        if not (terminated or truncated):
            continue

        # ---- a game just ended ----
        batch_returns.extend(discounted_returns(ep_rewards, GAMMA, args.point_returns))
        ep_rewards = []
        games_in_batch += 1
        total_rewards.append(ep_reward_sum)
        mean_reward = float(np.mean(total_rewards[-100:]))
        speed = (frame_idx - ts_frame) / (time.time() - ts)
        ts, ts_frame = time.time(), frame_idx
        hours = (time.time() - start) / 3600
        print("%d: done %d games, reward %.0f, mean reward %.3f, speed %.2f f/s, %.2f h" % (
            frame_idx, len(total_rewards), ep_reward_sum, mean_reward, speed, hours), flush=True)
        writer.add_scalar("reward", ep_reward_sum, frame_idx)
        writer.add_scalar("reward_100", mean_reward, frame_idx)
        writer.add_scalar("speed", speed, frame_idx)
        ep_reward_sum = 0.0
        state, _ = env.reset()
        if best_mean is None or mean_reward > best_mean:
            torch.save(net.state_dict(), "%s-%s-best.dat" % (args.env, tag))
            if best_mean is not None:
                print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean, mean_reward))
            best_mean = mean_reward
        if mean_reward > args.reward:
            stop_reason = "solved"
        elif args.max_hours > 0 and hours >= args.max_hours:
            stop_reason = "time"
        if games_in_batch < args.games_per_update:
            continue

        # ---- REINFORCE update on all steps of the last few games ----
        returns = np.asarray(batch_returns, dtype=np.float32)
        raw_mean = float(returns.mean())
        if args.return_norm in ("mean", "meanstd"):
            returns = returns - returns.mean()
        if args.return_norm == "meanstd":
            returns = returns / (returns.std() + 1e-8)
        n = len(returns)
        states_np = np.asarray(batch_states)
        actions_t = torch.tensor(batch_actions, dtype=torch.long)
        returns_t = torch.from_numpy(returns)

        optimizer.zero_grad()
        ent_sum, pol_sum = 0.0, 0.0
        for i in range(0, n, CHUNK):
            s = torch.from_numpy(states_np[i:i + CHUNK]).to(device).float() / 255.0
            a = actions_t[i:i + CHUNK].to(device)
            g = returns_t[i:i + CHUNK].to(device)
            log_p = F.log_softmax(net(s), dim=1)
            log_p_a = log_p.gather(1, a.unsqueeze(1)).squeeze(1)
            entropy = -(log_p.exp() * log_p).sum(dim=1)
            pol_loss = -(g * log_p_a).sum() / n
            loss = pol_loss - args.entropy_beta * entropy.sum() / n
            loss.backward()
            ent_sum += entropy.sum().item()
            pol_sum += pol_loss.item()
        grad_l2 = torch.sqrt(sum((q.grad ** 2).sum() for q in net.parameters() if q.grad is not None)).item()
        if args.grad_clip > 0:
            nn_utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optimizer.step()
        update_idx += 1

        writer.add_scalar("entropy", ent_sum / n, frame_idx)        # ln 6 = 1.79 means fully random
        writer.add_scalar("loss_policy", pol_sum, frame_idx)
        writer.add_scalar("return_mean", raw_mean, frame_idx)
        writer.add_scalar("return_std", float(np.std(batch_returns)), frame_idx)
        writer.add_scalar("grad_l2", grad_l2, frame_idx)
        writer.add_scalar("batch_steps", n, frame_idx)
        writer.add_scalar("updates", update_idx, frame_idx)
        batch_states, batch_actions, batch_returns = [], [], []
        games_in_batch = 0

    if stop_reason == "solved":
        print("Solved in %d frames!" % frame_idx)
    else:
        print("Stopped: time limit of %.2f h reached after %d frames, best mean reward %.3f" % (
            args.max_hours, frame_idx, best_mean))
    writer.close()
