# NoisyNet (Step 2a) and NoisyNet + PER (Step 3a)

**Contributor:** Sankalp Wahane
**Scripts:** `2a_noisynet_dqn.py` (NoisyNet), `3a_noisynet_per_dqn.py` (NoisyNet + PER)
**Hardware:** Google Colab, Tesla T4 (same GPU as the baseline)

## Step 2a: NoisyNet

**Algorithm.** Fortunato et al., *Noisy Networks for Exploration*, ICLR 2018. Epsilon-greedy is removed. The two fully connected layers become `NoisyLinear` layers whose weights are `μ + σ·ε`: both μ and σ are learned, and ε is fresh Gaussian noise on every forward pass (independent noise, σ₀ = 0.017, the paper's DQN setting; also used in Lapan, *Deep RL Hands-On*, Ch. 7). The agent always takes the argmax, and exploration comes from the noise. The network learns to shrink σ where it is confident, which shows up as a rising signal-to-noise ratio (`snr_1`, `snr_2` in TensorBoard).

### Run 1: stalled (`logs/2a_noisynet_run1_stalled.log`)
All baseline hyperparameters, with epsilon-greedy replaced by noisy layers and nothing else changed.

| | Value |
|---|---|
| Stopped at | 0.952 hr, 345,409 frames, 404 games |
| Mean reward (100 games) | **-20.58**, essentially no learning (baseline at 300k frames: -9.4) |
| Noise SNR | 1.04 / 3.46, barely changed |

**Diagnosis.** A freshly initialized NoisyNet is almost deterministic. The noise moves Q-values by about ±0.03, while the gap between actions' Q-values is about 0.15. Measured over 1,500 frames from four random initializations: one chose RIGHTFIRE 1,499 times out of 1,500, and another chose RIGHT 77% of the time. During the 10k-frame replay warm-up, the baseline (ε ≈ 1) fills its buffer with all 6 actions, but run 1's buffer held almost one action. Q-learning only updates Q(s,a) for actions present in the buffer, so the agent never learned that moving the paddle helps. It lost nearly every game 21–0 (about 855 frames per game).

**Fix.** Uniform random actions during the 10k-frame replay warm-up only (before any training). This is the original DQN's "replay start size" (Mnih et al., 2015). After warm-up, no epsilon is used. It also gives the same kind of starting data as the baseline. Verified locally: the warm-up buffer went from {RIGHTFIRE: 1499, LEFTFIRE: 1} to about 250 of each of the 6 actions.

### Run 2: solved (`logs/2a_noisynet.log`)
Random warm-up, plus training every 2 frames with batch 64 (the same number of samples per frame as the baseline's 1 × 32).

| | Baseline (ε-greedy) | **NoisyNet** | Change |
|---|---|---|---|
| Frames to 100-game mean ≥ 19 | 1,507,440 | **385,176** | **-74.4%** |
| Time to 19 | 3.958 hr | **1.057 hr** | **-73.3% (3.74× faster)** |
| Mean reward at convergence | 19.03 | **19.11** | higher |
| Speed | ~106 f/s | ~101 f/s | slightly slower per frame |

The per-frame speed is no better than the baseline's, so **the whole speedup comes from needing about 4× fewer frames**, i.e. better exploration. `snr_2` rose from 2.6 to about 17 by 225k frames as the network turned its own exploration down. The trained model (noise-free mean weights) won a full game **21–0** (`videos/2a_noisynet_gameplay_21-0.mp4`).

## Step 3a: NoisyNet + PER (`logs/3a_noisynet_per.log`)

**Algorithm.** Schaul et al., *Prioritized Experience Replay*, ICLR 2016, proportional variant. Transitions are sampled with probability ∝ (|TD error| + 1e-5)^α with α = 0.6. Importance-sampling weights (N·P(i))^-β, normalized by the max, correct the bias, with β annealed from 0.4 to 1.0 over 100k frames. Sampling and priority updates use a **sum tree**, as proposed in the PER paper. Everything else is identical to NoisyNet run 2.

| | NoisyNet | **NoisyNet + PER** | Change |
|---|---|---|---|
| Frames to 19 | 385,176 | 580,539 | **+50.7%** |
| Time to 19 | 1.057 hr | 1.813 hr | **+71.5%** |
| Mean reward at convergence | 19.11 | 19.02 | about the same |
| Speed | ~101 f/s | ~89 f/s | about 12% slower per frame |

**Result: PER did not improve NoisyNet in this setup.** It is still far faster than the baseline (61.5% fewer frames, 54.2% less time). Likely reasons:
1. **Tiny replay buffer.** We keep the book's 10k buffer, while the PER paper used 1M. With training every 2 frames at batch 64, uniform sampling already replays every transition about 32 times before it leaves the buffer, so there are few neglected important transitions for PER to find.
2. **Pong has dense rewards.** Every point gives a reward, and PER helps most when informative transitions are rare.
3. **PER and noisy layers interact.** Priorities are |TD error| computed through noisy weights, so part of each priority is noise. Also, with PER the noise SNR rose faster early (`snr_2` about 13 by 80k frames vs about 9.4 at 87k for NoisyNet), i.e. exploration was turned down sooner, and the steep climb started about 100–150k frames later.
4. **Per-frame overhead.** Tree operations and a GPU→CPU copy of TD errors every step cost about 12% speed.
5. **Single-run variance.** The book author's identical basic DQN runs took 87 vs 163 min.

For contrast, PER on the other Step 2 algorithm (UCB Q-ensemble, Step 3b) gave 4.4% fewer frames. See `UCB.md` and the main README.

## Files
- `logs/`: full training logs (each ends with `Solved in … frames!`, except the stalled run)
- `tensorboard/`: event files; view with `tensorboard --logdir results/tensorboard`
- `plots/2a_noisynet/`, `plots/2a_noisynet_run1_stalled/`, `plots/3a_noisynet_per/`: TensorBoard screenshots and full-run graphs drawn from the event files (the Colab TensorBoard view froze for the PER run)
- `videos/2a_noisynet_gameplay_21-0.mp4`: trained NoisyNet playing one full game

## References
- Fortunato et al. (2018). *Noisy Networks for Exploration.* ICLR.
- Schaul, Quan, Antonoglou & Silver (2016). *Prioritized Experience Replay.* ICLR.
- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- Lapan, M. *Deep Reinforcement Learning Hands-On*, Chapters 6–7. Packt.
