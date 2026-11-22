#!/usr/bin/env python3
# ============================================================================
# Project 2: A2C (Advantage Actor-Critic) on Atari Pong.
#
# A port of the book's Chapter 10 code (02_pong_a2c.py) with the same default
# hyperparameters: 50 games in parallel, 4-step returns, batch 128,
# learning rate 0.001 (Adam, eps 1e-3), entropy beta 0.01, gradient clip 0.1.
# Changes, so the settings match Project 1: the Project 1 wrappers
# (lib/wrappers.py, gymnasium) instead of the old ptan/gym ones, and the
# stop bound 19.0 instead of the book's 18. The book's ptan experience source
# is replaced by a few lines below that do the same job (4-step "first-last" transitions).
#
# Plain English: the network has two outputs. The actor picks buttons. The critic
# guesses how well the game is going. After each move the actor is pushed toward
# moves that turned out BETTER than the critic expected (the "advantage").
# ============================================================================
import argparse
import collections
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim
from tensorboardX import SummaryWriter

from lib import wrappers
from lib.pg_models import ActorCriticNet

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.0
GAMMA = 0.99
LOG_EVERY = 20  # updates between TensorBoard points for the loss/advantage curves


def to_uint8(state):
    return np.rint(state * 255.0).astype(np.uint8)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", action="store_true", help="Train on the GPU")
    p.add_argument("--env", default=DEFAULT_ENV_NAME)
    p.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND, help="Stop when the 100-game mean exceeds this")
    p.add_argument("--max-hours", type=float, default=0.0, help="Stop after this many hours (0 = no limit)")
    p.add_argument("--lr", type=float, default=0.001, help="Adam learning rate (book: 0.001)")
    p.add_argument("--envs", type=int, default=50, help="Games played in parallel (book: 50)")
    p.add_argument("--batch-size", type=int, default=128, help="Transitions per update (book: 128)")
    p.add_argument("--reward-steps", type=int, default=4, help="n in the n-step return (book: 4)")
    p.add_argument("--entropy-beta", type=float, default=0.01, help="Entropy bonus weight (book: 0.01)")
    p.add_argument("--grad-clip", type=float, default=0.1, help="Gradient norm clip (book: 0.1)")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    tag = "a2c-lr%g-e%d-b%d-n%d-ent%g" % (args.lr, args.envs, args.batch_size, args.reward_steps, args.entropy_beta)
    print("Config:", tag, vars(args), flush=True)

    envs = [wrappers.make_env(args.env) for _ in range(args.envs)]
    states = []
    for i, e in enumerate(envs):
        s, _ = e.reset(seed=None if args.seed is None else args.seed + i)
        states.append(s)
    n_actions = envs[0].action_space.n
    net = ActorCriticNet(envs[0].observation_space.shape, n_actions).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, eps=1e-3)
    writer = SummaryWriter(comment="-%s-%s" % (args.env, tag))
    print(net)

    histories = [collections.deque() for _ in envs]   # last n (state, action, reward) per game
    game_rewards = [0.0] * len(envs)
    batch = []    # (first_state_uint8, action, discounted n-step reward, last_state_uint8 or None)
    total_rewards, best_mean = [], None
    frame_idx, update_idx = 0, 0
    start = ts = time.time()
    ts_frame = 0
    track = collections.defaultdict(list)
    stop_reason = None

    def emit(hist, last_state):
        disc = 0.0
        for k, (_, _, r) in enumerate(hist):
            disc += (GAMMA ** k) * r
        s0, a0, _ = hist[0]
        batch.append((s0, a0, disc, None if last_state is None else to_uint8(last_state)))

    while stop_reason is None:
        # ---- every game takes one step; one forward pass picks all 50 buttons ----
        with torch.no_grad():
            logits, _ = net(torch.from_numpy(np.asarray(states)).to(device))
            actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        for i, e in enumerate(envs):
            new_s, r, terminated, truncated, _ = e.step(int(actions[i]))
            done = terminated or truncated
            frame_idx += 1
            game_rewards[i] += r
            hist = histories[i]
            hist.append((to_uint8(states[i]), int(actions[i]), r))
            if len(hist) == args.reward_steps:
                emit(hist, None if done else new_s)
                hist.popleft()
            if done:
                while hist:                      # game over: the last few steps get shorter returns, no bootstrap
                    emit(hist, None)
                    hist.popleft()
                total_rewards.append(game_rewards[i])
                mean_reward = float(np.mean(total_rewards[-100:]))
                speed = (frame_idx - ts_frame) / (time.time() - ts)
                ts, ts_frame = time.time(), frame_idx
                hours = (time.time() - start) / 3600
                print("%d: done %d games, reward %.0f, mean reward %.3f, speed %.2f f/s, %.2f h" % (
                    frame_idx, len(total_rewards), game_rewards[i], mean_reward, speed, hours), flush=True)
                writer.add_scalar("reward", game_rewards[i], frame_idx)
                writer.add_scalar("reward_100", mean_reward, frame_idx)
                writer.add_scalar("speed", speed, frame_idx)
                game_rewards[i] = 0.0
                new_s, _ = e.reset()
                if best_mean is None or mean_reward > best_mean:
                    torch.save(net.state_dict(), "%s-%s-best.dat" % (args.env, tag))
                    if best_mean is not None:
                        print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean, mean_reward))
                    best_mean = mean_reward
                if len(total_rewards) >= 100 and mean_reward > args.reward:
                    stop_reason = "solved"
                elif args.max_hours > 0 and hours >= args.max_hours:
                    stop_reason = "time"
            states[i] = new_s
        if stop_reason is not None:
            break

        # ---- A2C updates, one per full batch ----
        while len(batch) >= args.batch_size:
            chunk, batch = batch[:args.batch_size], batch[args.batch_size:]
            states_v = torch.from_numpy(np.asarray([c[0] for c in chunk])).to(device).float() / 255.0
            actions_t = torch.tensor([c[1] for c in chunk], dtype=torch.long, device=device)
            ref = np.array([c[2] for c in chunk], dtype=np.float32)
            not_done = [j for j, c in enumerate(chunk) if c[3] is not None]
            if not_done:
                last_v = torch.from_numpy(np.asarray([chunk[j][3] for j in not_done])).to(device).float() / 255.0
                with torch.no_grad():
                    ref[not_done] += (GAMMA ** args.reward_steps) * net(last_v)[1].squeeze(-1).cpu().numpy()
            ref_v = torch.from_numpy(ref).to(device)   # target for the critic: n real rewards + critic's guess after that

            optimizer.zero_grad()
            logits_v, value_v = net(states_v)
            value_v = value_v.squeeze(-1)
            loss_value = F.mse_loss(value_v, ref_v)
            log_p = F.log_softmax(logits_v, dim=1)
            adv = ref_v - value_v.detach()              # advantage: better or worse than the critic expected
            loss_policy = -(adv * log_p.gather(1, actions_t.unsqueeze(1)).squeeze(1)).mean()
            entropy = -(log_p.exp() * log_p).sum(dim=1).mean()
            loss = loss_policy + loss_value - args.entropy_beta * entropy
            loss.backward()
            grad_l2 = torch.sqrt(sum((q.grad ** 2).sum() for q in net.parameters() if q.grad is not None)).item()
            nn_utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()
            update_idx += 1

            track["advantage"].append(adv.mean().item())
            track["values"].append(value_v.mean().item())       # critic's prediction
            track["batch_rewards"].append(ref_v.mean().item())  # what it was trained toward
            track["entropy"].append(entropy.item())
            track["loss_policy"].append(loss_policy.item())
            track["loss_value"].append(loss_value.item())
            track["grad_l2"].append(grad_l2)
            if update_idx % LOG_EVERY == 0:
                for k, v in track.items():
                    writer.add_scalar(k, float(np.mean(v)), frame_idx)
                writer.add_scalar("updates", update_idx, frame_idx)
                track.clear()

    if stop_reason == "solved":
        print("Solved in %d frames!" % frame_idx)
    else:
        print("Stopped: time limit of %.2f h reached after %d frames, best mean reward %.3f" % (
            args.max_hours, frame_idx, best_mean))
    writer.close()
