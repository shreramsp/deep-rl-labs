# UCB Q-ensemble (Step 2b) and UCB + PER (Step 3b)

**Contributor:** Shreram Palanisamy
**Scripts:** `experiments/2x_bootstrapped_dqn.py` (first attempt), `2b_ucb_ensemble_dqn.py`, `3b_ucb_per_dqn.py`
**Hardware:** Google Colab, Tesla T4 (same GPU as the baseline)

## First attempt: Bootstrapped DQN (`experiments/2x_bootstrapped_dqn.py`)

**Algorithm.** Osband et al., *Deep Exploration via Bootstrapped DQN*, NeurIPS 2016. One shared convolutional body with K Q-value "heads". At the start of each game one head is chosen at random and followed greedily for the whole game ("deep exploration"). Paper settings: p = 1 (every head trains on every sample) and the shared-body gradient scaled by 1/K. Each head has its own target head. The heads are stored as stacked tensors, so all K run in one batched matrix multiply.

| Run | Setting | Outcome |
|---|---|---|
| Untuned (`logs/2x_bootstrap_run1_untuned.log`) | K = 10, train every frame, batch 32 | Learned at the baseline's per-frame rate, but ran at only **~60 f/s** (vs ~110) because 10 heads are updated every step. Projected 6–7 hr, so it was stopped at 0.68 hr. |
| Tuned (`logs/2x_bootstrap_tuned.log`) | K = 5, train every 2 frames, batch 64 | Back to ~110 f/s. Reached **16.7 at 1.03 hr** (the baseline needed ~3 hr), best **18.64**, then **plateaued at 17.75–18.64 for ~1.9 hr**. Stopped at 3.014 hr (1,197,825 frames, 18.50), **not solved**. |

**Why it plateaued (diagnosis).** Each game is played by one randomly chosen head. The per-game reward graph (`plots/2x_bootstrap_tuned/reward_full_3.01hr.png`) shows games mostly at 18–21 with frequent dips to ~10 and occasionally to -8 or -15: weaker heads keep dragging the 100-game mean below 19. Osband et al. note that individual heads differ and that a vote across heads often beats any single head. That motivated switching to an algorithm that **acts with all heads at once**.

## Step 2b: UCB exploration with Q-ensembles (`2b_ucb_ensemble_dqn.py`)

**Algorithm.** Chen, Sidor, Abbeel & Schulman, *UCB Exploration via Q-Ensembles* (2017), Algorithm 2. The network, loss, and per-head targets are the same as Bootstrapped DQN, but on **every step** the agent asks all heads:

  a = argmax_a [ mean_k Q_k(s,a) + λ · std_k Q_k(s,a) ],   λ = 0.1 (paper value)

When the heads agree, the agent exploits their mean. When they disagree (high std), the action gets an optimism bonus and is tried. This is the Upper Confidence Bound idea (Sutton & Barto §2.7), with the ensemble's spread as the uncertainty. In the paper, UCB had the best maximal mean reward on 30 of 49 Atari games and beat both Double DQN and Bootstrapped DQN. Settings: K = 5, train every 2 frames, batch 64 (the same as tuned Bootstrapped DQN, so **the only difference is the action-selection rule**), plus uniform random actions during the 10k-frame replay warm-up (the original DQN's "replay start size", Mnih et al. 2015; also in Chen et al.'s hyperparameter table).

| | Baseline (ε-greedy) | **UCB Q-ensemble** | Change |
|---|---|---|---|
| Frames to 100-game mean ≥ 19 | 1,507,440 | **568,001** | **-62.3%** |
| Time to 19 | 3.958 hr | **1.438 hr** | **-63.7% (2.75× faster)** |
| Mean reward at convergence | 19.03 | 19.03 | same |
| Speed | ~106 f/s | ~110 f/s | same per frame |

Between 0.58 and 0.92 hr it climbed faster than any other run (-9.7 → +13.1). **Unlike Bootstrapped DQN, with the same network and settings, it did not plateau.** That supports the diagnosis that acting with one random head per game was the problem.

## Step 3b: UCB + PER (`3b_ucb_per_dqn.py`)

**Algorithm.** Schaul et al., *Prioritized Experience Replay* (ICLR 2016), proportional variant, added to the UCB agent unchanged: α = 0.6, importance-sampling β annealed 0.4 → 1.0 over 100k frames, priority = |TD error| + 1e-5 averaged over the K heads, and a sum tree for O(log N) sampling. The PER buffer implementation is the same one used in Step 3a.

| | UCB | **UCB + PER** | Change |
|---|---|---|---|
| Frames to 19 | 568,001 | **542,750** | **-4.4%** ✅ |
| Time to 19 | 1.438 hr | 1.517 hr | +5.5% |
| Mean reward at convergence | 19.03 | 19.01 | same |
| Speed | ~110 f/s | ~99 f/s | about 10% slower per frame (PER overhead) |

**Where PER helped.** In the middle of training UCB + PER was slightly behind UCB at the same frame count, but it **finished the last stretch faster** (18.53 vs 18.01 at ~495k frames) and crossed 19 about 25k frames earlier. Its per-game rewards after ~450k frames are visibly steadier. Intuition: once the agent wins most points, the few points it still loses are the high-TD-error ("surprising") transitions, and PER replays exactly those more often.

**Honest caveats.** A 4.4% difference from a single run is within run-to-run variance. The improvement is in sample efficiency, not wall-clock time, because PER's per-frame overhead outweighs the frame savings. PER's benefit is also limited by the book's 10k replay buffer (the PER paper used 1M): with training every 2 frames at batch 64, uniform sampling already replays each transition ~32 times.

## Files
- `logs/2x_bootstrap_run1_untuned.log`, `logs/2x_bootstrap_tuned.log`, `logs/2b_ucb.log`, `logs/3b_ucb_per.log`
- `tensorboard/2x_bootstrap_*`, `tensorboard/2b_ucb/`, `tensorboard/3b_ucb_per/`
- `plots/2x_bootstrap_tuned/` (full-run graphs drawn from the event file, since the Colab TensorBoard view froze), `plots/2x_bootstrap_run1_untuned/`, `plots/2b_ucb/`, `plots/3b_ucb_per/` (incl. `step3_ucb_vs_ucb_per.png` and the β schedule)

## References
- Chen, Sidor, Abbeel & Schulman (2017). *UCB Exploration via Q-Ensembles.* arXiv:1706.01502.
- Osband, Blundell, Pritzel & Van Roy (2016). *Deep Exploration via Bootstrapped DQN.* NeurIPS.
- Schaul, Quan, Antonoglou & Silver (2016). *Prioritized Experience Replay.* ICLR.
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*, §2.7.
- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
