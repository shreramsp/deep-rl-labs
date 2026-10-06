# CMPE 260 Project 1 (DQN Pong): Results Log (lab notebook)

> Chronological record of every run, written as the runs happened. For the summary, see the main `README.md`. All paths are relative to the `pong-dqn/` folder.

Running record of every training run's key numbers, extracted from TensorBoard / training.log at the time each run finished. Use this as the single source of truth when writing the report — no need to dig back through Colab.

## Baseline (vanilla DQN: replay buffer + target network + epsilon-greedy)

| Metric | Value |
|---|---|
| Run name | `Oct02_22-06-02_5eeed8b9b8a3-PongNoFrameskip-v4` |
| GPU | Tesla T4 |
| Stop condition | Auto-stopped (`MEAN_REWARD_BOUND = 19.0`) |
| Frames to converge | 1,507,440 |
| Games played | 670 |
| Time to converge | 3.958 hours (~3 hr 57 min) |
| Final mean reward (100-game rolling avg) | **19.03** |
| Log confirmation | `Solved in 1507440 frames!` |

**Notes:** This run completed cleanly and exited on its own (unlike the first baseline attempt, which had to be manually killed and showed a text-log/TensorBoard mismatch due to stdout buffering). All four TensorBoard panels (reward, reward_100, epsilon, speed) agree exactly on the same endpoint — step 1,507,440, 3.958 hr — so this is a fully consistent, reliable baseline number. Use **19.03 / 1,507,440 frames / 3.958 hr / Tesla T4** as the official baseline for all Step 2 comparisons.

Screenshots: `results/plots/1_baseline/` (training_log_solved.png, epsilon.png, reward.png, reward_100.png, speed.png)

Earlier (superseded) run: 19.32 mean reward / 1,247,055 frames / 3.199 hr — kept for reference in (superseded; not included), not used for reporting.

### Baseline reference curve (rebuilt from `results/logs/1_baseline.log`; rebuilt total = 3.959 hr, matches TensorBoard)

Use these to judge Step 2 runs at the same wall-clock time or the same frame count.

| Time | Frame | Mean reward | | Frame | Time | Mean reward |
|---|---|---|---|---|---|---|
| 16.4 min | 117,383 | -18.42 | | 68,000 | 8.9 min | -19.89 |
| 20 min | 140,161 | -17.56 | | 130,000 | 18.3 min | -17.96 |
| 30 min | 204,306 | -15.06 | | 200,000 | 29.6 min | -15.18 |
| 45 min | 301,260 | -9.36 | | 300,000 | 45.1 min | -9.36 |
| 60 min | 393,959 | -0.90 | | 400,000 | 60.6 min | -0.33 |
| 90 min | 584,547 | 12.20 | | 600,000 | 92.4 min | 12.69 |
| 120 min | 768,943 | 14.43 | | 800,000 | 125.0 min | 14.18 |
| 180 min | 1,148,276 | 16.45 | | 1,000,000 | 157.0 min | 16.11 |
| | | | | 1,200,000 | 188.4 min | 16.75 |

Baseline speed: ~850 f/s for the first 10k frames (no training yet), then ~105–115 f/s once training starts. The training step is the bottleneck.

**Time budget math:** to beat 3.56 hr, an algorithm running at S f/s can afford about 3.56 × 3600 × S frames. At 100 f/s that is about 1.28M frames (15% fewer than the baseline's 1.507M). At 60 f/s it is about 0.77M frames (49% fewer).

## Step 2 — Algorithm 1: NoisyNet (`2a_noisynet_dqn.py`)

### Run 1: untuned, no random warm-up: STALLED, stopped at 0.95 hr

All baseline hyperparameters, epsilon-greedy replaced by NoisyLinear (sigma_0 = 0.017). Run name `Oct04_00-43-38_37860bb5d487-PongNoFrameskip-v4-noisy`.

| Metric | Value |
|---|---|
| GPU | (confirm from nvidia-smi — T4 expected) |
| @ 25.73 min | 160,965 frames, mean reward -20.53, ~100–104 f/s |
| @ 45.82 min | **279,349 frames, mean reward -20.60**, essentially no learning |
| Baseline at the same time / frames | -9.36 at 45 min / 300k frames. Book author's NoisyNet: -3.18 at 300k frames |
| Final (log) @ 0.952 hr | **345,409 frames, 404 games, mean -20.58**, snr 1.04/3.46 (noise barely reduced). 404 games in 345k frames (~855 frames per game) means it was losing quickly every game |
| Decision | Stopped. Diagnosed (below) and rerun with a fix. |

Artifacts: `results/{logs,tensorboard}/2a_noisynet_run1_stalled*` (model, `training_noisy.log`, TensorBoard events).
Screenshots: `results/plots/2a_noisynet_run1_stalled/` (noisynet_untuned_25min_reward100.png, noisynet_untuned_46min_reward100_STALLED.png).

**Per-game rewards (from the TensorBoard `reward` graph):** around 20–25 min, almost every single game ended **-21** (a 21–0 loss), not just the 100-game mean. Run 1 averaged about 855 frames per game, about the minimum length of a 21–0 game, so the agent was losing every point as fast as possible.

**Diagnosis (for the report).** A freshly initialized NoisyNet is close to deterministic. The noise moves Q-values by about ±0.03, but the gap between actions' Q-values is about 0.15. Measured locally over 1,500 frames from 4 random initializations: one init chose RIGHTFIRE 1,499 of 1,500 times, and another chose RIGHT 77% of the time. During the first 10k frames (replay warm-up, before training), the baseline's epsilon is about 1, so its buffer gets all 6 actions uniformly. Run 1's buffer got almost a single action. Q-learning only updates Q(s,a) for actions that appear in the buffer, so the other actions were never learned and the agent stayed stuck (~0.4 points per game).

**Fix:** uniform random actions during the 10k-frame replay warm-up only. This is the original DQN's "replay start size" (Mnih et al., 2015, which defines it as the number of uniform random actions before learning starts), also used by Dopamine's agents. After warm-up, there is no epsilon, and all exploration comes from the noisy layers. It also makes the warm-up data identical in kind to the baseline's. Verified locally: the warm-up buffer went from {RIGHTFIRE: 1499, LEFTFIRE: 1} to about 250 of each of the 6 actions.

### Run 2: random warm-up + tuned (train every 2 frames, batch 64)

Command: `python 2a_noisynet_dqn.py --cuda --train-every 2 --batch-size 64`. Run tag `-noisy-te2-b64-warmup`. Same speed tuning as Bootstrapped DQN run 2, so both Step 2 algorithms are compared under identical settings. (`--no-random-warmup` reproduces run 1.)

Interim log: @ 0.211 hr (12.7 min): 86,859 frames, 73 games, mean -19.74, snr 1.59/9.43, ~105 f/s. Baseline at 87k frames: -19.51, so roughly on par, and far better than run 1 (-20.53 at 161k). snr_2 rose from 2.6 to 9.4, meaning the network is cutting its own output-layer noise.

**RESULT: SOLVED.** Run name `Oct04_01-41-06_37860bb5d487-PongNoFrameskip-v4-noisy-te2-b64-warmup`. Log confirmation: `Solved in 385176 frames! (1.057 hr)`.

| Metric | Value | vs baseline (1,507,440 frames / 3.958 hr / 19.03) |
|---|---|---|
| GPU | **Tesla T4** (nvidia-smi screenshot) | same GPU ✅ |
| Speed | ~101 f/s average (98–105) | about 8% slower per frame |
| Frames to converge | **385,176** (~208 games) | **74.4% fewer frames** (baseline: 670 games) |
| Time to converge | **1.057 hr (63.4 min)** | **73.3% less time, 3.74× faster** (target was ≤ 3.56 hr) ✅ |
| Final mean reward | **19.11** | higher than 19.03 ✅ |

Progress (from the monitor log): 0.368 hr 145k frames -18.41 · 0.540 hr 205k -15.41 · 0.709 hr 265k **-2.81** · 0.876 hr 322k **+9.30** · 1.042 hr 380k 18.37 · 1.057 hr 385k **19.11**. At the same 63 minutes, the baseline was only at about -0.9. The steep climb (-15 → +18) took about 175k frames / 30 min. The baseline's equivalent climb plus its 14 → 19 crawl took about 1.2M frames / 3 hr.

SNR (noise) curves: snr_1 rose from 1.05 to 2.25 and plateaued around 220k frames. snr_2 rose from 2.6 to a peak of about 17 at 225k (the network turning its output-layer noise down as play became confident), then settled around 13.7. Mean reward per game hovers around 19–21 from about 220k frames.

**Honest caveats for the report:** (1) This is a single run, and single-run variance on Pong is large (the author's same algorithm took 87 vs 163 min). (2) This run has two changes besides NoisyNet: the random replay warm-up (which brings the warm-up data in line with the baseline's ε ≈ 1 start) and train-every-2 / batch-64. The per-frame speed is no better than the baseline (~101 vs ~110 f/s), so **the whole speedup comes from needing about 4× fewer frames**, which is the exploration effect, not faster computation. (3) For reference, the book author's NoisyNet solved (bound 18) in 571k frames.

Screenshots: `results/plots/2a_noisynet/` (elapsed_hours, reward, reward_100, snr_1, snr_2, speed, training_log_solved, nvidia_smi).
Artifacts: `results/{logs,tensorboard}/2a_noisynet*` (model `.dat`, `training_noisy_tuned.log`, `runs/` TensorBoard events, and `noisynet_gameplay_21-0.mp4`).

**Gameplay check:** `play.py` with the best model (noise-free mean network, the standard greedy NoisyNet evaluation) won one full game **21–0**. It used all 6 actions (NOOP 478, LEFT 424, LEFTFIRE 422, FIRE 189, RIGHTFIRE 127, RIGHT 53). Video: `results/videos/2a_noisynet_gameplay_21-0.mp4`.

## Step 2 — Algorithm 2: Bootstrapped DQN (`experiments/2x_bootstrapped_dqn.py`)

### Run 1: untuned (K = 10 heads, train every frame, batch 32): STOPPED EARLY on purpose

Run name `Oct04_00-44-57_124e0ce7bf21-PongNoFrameskip-v4-bootstrap`.

| Metric | Value |
|---|---|
| GPU | (confirm from nvidia-smi — T4 expected) |
| Last recorded @ 25.15 min | 105,488 frames, mean reward **-18.62** (smoothed -18.71) |
| Speed | **~60–70 f/s** vs baseline ~110 f/s (10 heads ≈ 16M extra FC weights updated every step) |
| Per-frame learning | Same as baseline (baseline ≈ -18.7 at 105k frames) |
| Projection | At baseline-like frame count (~1.5M) and ~65 f/s: **~6–7 hr**. Beating 3.56 hr would need about 49% fewer frames than the baseline; it was tracking 0% fewer. |
| Final (log) | Actually stopped at **0.682 hr: 162,631 frames, 84 games, mean -16.24**, ~60 f/s |
| Decision | Stopped. The 10% speed target was mathematically out of reach at this per-frame cost. |

Screenshot: `results/plots/2x_bootstrap_run1_untuned/bootstrap_untuned_25min_reward100_STOPPED.png`.

**For the report:** Bootstrapped DQN explored about as well as epsilon-greedy per frame, but the K = 10 ensemble made each training step about 40% slower on a T4. The paper reports under 20% overhead, but its heads are a smaller share of the cost than with our tiny network and batch of 32. This motivated the tuning below, using the assignment's "and tune it" allowance.

### Run 2: tuned (K = 5 heads, train every 2 frames, batch 64)

Command: `python experiments/2x_bootstrapped_dqn.py --cuda --heads 5 --train-every 2 --batch-size 64`. TensorBoard run name and model file carry the tag `-bootstrap-k5-te2-b64`.

Rationale:
- **K = 5:** halves the extra head cost. Osband et al. (2016, Fig. 5a) studied sensitivity to K.
- **Train every 2 frames, batch 64:** the same number of samples learned per frame played, with half the gradient steps. The original DQN (Mnih et al., 2015) trained every 4 frames.

Interim log: @ 0.198 hr: 83,422 frames, -20.19 · @ 0.368 hr: 149,723 frames, -18.75 · @ 0.530 hr (31.8 min): **212,951 frames, 126 games, mean -16.05, ~110 f/s**. Last ~22 games average about -8.7 (reconstructed from the 100-game means) vs the baseline's -5.8 at the same frames (mean100 -14.54), so about 3 points behind per frame but climbing fast.

Progress (monitor log): 0.701 hr 281k frames -5.27 · 0.868 hr 346k +6.99 · 1.034 hr 412k **16.71** · 1.199 hr 477k 18.31 · 1.371 hr 546k 18.59 · 1.536 hr 612k 18.14 · 1.705 hr 679k 17.83 (374 games) · 1.867 hr 744k 18.28 · 2.033 hr 809k 18.43 · **2.196 hr 872k 18.50** (479 games) · 2.365 hr 940k 18.36 · 2.370 hr 942k 18.41 (516 games) · 2.531 hr 1.006M 17.82 · 2.700 hr 1.073M 17.81 · **2.866 hr 1.138M 18.20 (622 games)**.

**FINAL: STOPPED at 3.014 hr, NOT SOLVED.** Last log line: **1,197,825 frames, 655 games, mean 18.50, 3.014 hr**. Best 100-game mean: **18.64 at 880,995 frames / 2.219 hr** (from the log's "Best mean reward updated" lines; the 10-minute monitor snapshots had only caught 18.59 at 1.371 hr). From about 420k frames (~1.1 hr) it held between 17.63 and 18.64 for about 1.9 hr and never crossed 19.0. It was stopped about 0.55 hr before the 3.56 hr cutoff, since a +0.5 gain after that long a plateau was implausible. Speed averaged ~110 f/s (same as the baseline).

**Why it plateaued (likely):** each game is played by one randomly chosen head, so weaker heads pull down the 100-game mean. The full-run `reward` graph shows frequent single-game dips to about 10, and occasionally to -8 or -15, among games at 18–21. Osband et al. note that individual heads vary and that an ensemble vote often beats any single head.

**Report summary:** Bootstrapped DQN learned much faster early (16.7 at 1.03 hr; the baseline needed about 3 hr to reach that), and at the same elapsed time it stayed well ahead (at 2 hr: about 18.4 vs about 14.4). But it plateaued below 19 and did **not** meet the convergence target. It was replaced as Algorithm B by UCB Q-ensembles.

Artifacts: `results/{logs,tensorboard}/2x_bootstrap_tuned*` (tuned run: model, log, events) and `results/{logs,tensorboard}/2x_bootstrap_run1_untuned*` (untuned K = 10 run).
Graphs (drawn from the full event file with `tools/plot_runs.py`, since the Colab TensorBoard view had stopped refreshing at 1.56 hr): `results/plots/2x_bootstrap_tuned/{reward,reward_100,speed,elapsed_hours}_full_3.01hr.png`. The earlier Colab screenshots `*_upto_1.56hr.png` cover only the first 1.56 hr.

| Metric | Value |
|---|---|
| GPU | |
| Speed | ~108–110 f/s (was 60–70 untuned) |
| Frames to converge | |
| Time to converge | |
| Final mean reward | |
| Target (≥10% faster than baseline) | ≤ 3.56 hours |
| Target (score closer to 21) | > 19.03 |

## Step 2 — Algorithm B (replacement): UCB exploration with Q-ensembles (`2b_ucb_ensemble_dqn.py`)

**Why it replaced Bootstrapped DQN:** tuned Bootstrapped DQN plateaued at 18.1–18.6 for over an hour (18.41 at 2.37 hr) and looked unlikely to reach 19 by 3.56 hr. It plays each game with one randomly chosen head, so weaker heads keep dragging the 100-game mean down.

**Algorithm:** Chen, Sidor, Abbeel & Schulman, *UCB Exploration via Q-Ensembles* (2017, arXiv:1706.01502), Algorithm 2. Same K-head ensemble as Bootstrapped DQN (shared conv body, per-head targets, p = 1, conv gradient / K), but on **every step** the agent uses all heads: `a = argmax_a [mean_k Q_k(s,a) + λ·std_k Q_k(s,a)]` with **λ = 0.1** (paper value). Agreement means exploit; disagreement adds an optimism bonus. Paper results: ensemble voting beats Double DQN and Bootstrapped DQN, UCB improves further, and UCB has the best max mean reward on 30 of 49 Atari games (Pong: 20.95). Random replay warm-up (10k frames, uniform actions) as for NoisyNet; Chen et al.'s hyperparameter table lists the same "uniform random actions before learning starts".

Command: `python 2b_ucb_ensemble_dqn.py --cuda --heads 5 --train-every 2 --batch-size 64`. Run tag `-ucb-k5-l0.1-te2-b64-warmup`. These are the same K and speed settings as tuned Bootstrapped DQN, so **the only difference between the two is the action-selection rule**.

**RESULT: SOLVED.** Run name `Oct04_04-17-50_9b359875ce08-PongNoFrameskip-v4-ucb-k5-l0.1-te2-b64-warmup`. Log confirmation: `Solved in 568001 frames! (1.438 hr)`.

| Metric | Value | vs baseline (1,507,440 frames / 3.958 hr / 19.03) |
|---|---|---|
| GPU | **Tesla T4** (nvidia-smi screenshot) | same GPU ✅ |
| Speed | ~110 f/s average (103–117) | about the same per frame |
| Frames to converge | **568,001** (~318 games) | **62.3% fewer frames** |
| Time to converge | **1.438 hr (86.3 min)** | **63.7% less time, 2.75× faster** (target ≤ 3.56 hr) ✅ |
| Final mean reward | **19.03** | equal to the baseline (both stop right at the 19.0 bound) |

Progress (monitor log): 0.251 hr 103k -19.87 · 0.420 hr 166k -18.03 · 0.584 hr 229k **-9.69** · 0.752 hr 294k **+1.61** · 0.919 hr 360k **13.11** · 1.082 hr 425k 16.93 · 1.247 hr 493k 18.01 · 1.417 hr 560k 18.86 · **1.438 hr 568k 19.03 (solved)**. Between 0.58 and 0.92 hr it climbed faster than any other run. From about 190k frames, single-game rewards swung between 10 and 21, with rare dips to about -6 and -19 (the UCB bonus still exploring where heads disagree). Those dips faded from about 450k frames, and it crossed 19 at 568k. **Unlike Bootstrapped DQN (same 5-head network, same settings), it did NOT plateau.** That supports the diagnosis that acting with one random head per game was Bootstrapped DQN's problem, and acting on all heads via UCB fixes it.

Screenshots: `results/plots/2b_ucb/` (training_log_solved, nvidia_smi, elapsed_hours, reward, reward_100, speed).
Artifacts: `results/{logs,tensorboard}/2b_ucb*` (model, `training_ucb.log`, `runs/` events).

## Step 2 summary table

| | Baseline (ε-greedy) | **NoisyNet** | **UCB Q-ensemble** | Bootstrapped DQN (tuned) |
|---|---|---|---|---|
| Script | `1_baseline_dqn.py` | `2a_noisynet_dqn.py` | `2b_ucb_ensemble_dqn.py` | `experiments/2x_bootstrapped_dqn.py` |
| Exploration | random action with prob ε (1.0 → 0.02 over 100k frames) | learned weight noise | argmax mean + 0.1·std over 5 heads | 1 random head per game |
| Settings | book defaults | + random warm-up, train every 2, batch 64 | + random warm-up, 5 heads, train every 2, batch 64 | 5 heads, train every 2, batch 64 |
| GPU | T4 | T4 | T4 | T4 |
| Speed (avg f/s) | ~106 | ~101 | ~110 | ~110 |
| Frames to 19 | 1,507,440 | **385,176 (-74.4%)** | **568,001 (-62.3%)** | not reached (1.198M, best 18.64) |
| Time to 19 | 3.958 hr | **1.057 hr (-73.3%, 3.74×)** | **1.438 hr (-63.7%, 2.75×)** | not reached (stopped at 3.014 hr) |
| Mean reward at convergence | 19.03 | **19.11** | 19.03 | — |
| ≥10% faster (≤ 3.56 hr)? | — | ✅ | ✅ | ❌ |

**Step 3 base: NoisyNet** (fastest in both time and frames, highest score). UCB is the second qualifying Step 2 algorithm.

## Step 2 comparison graphs

`results/plots/comparison/reward_100_vs_time.png` (x = wall-clock hours on a T4, with the 3.56 hr target line) and `reward_100_vs_frames.png`: baseline vs NoisyNet vs Bootstrapped DQN (tuned), drawn from each run's TensorBoard event file by `tools/plot_runs.py` . Final values: baseline 19.03 at 3.96 hr / 1,507,440 frames (solved); NoisyNet 19.11 at 1.06 hr / 385,176 frames (solved); Bootstrapped DQN 18.48 at 3.00 hr / 1,190,853 frames (stopped; the event file ends slightly before the last log line). Add UCB once it finishes.

### Speed note (Step 2 run 2s)
`--train-every 2 --batch-size 64` fixed Bootstrapped DQN's slowdown (60 → 110 f/s), but neither algorithm became faster than the baseline (~105–110 f/s). The training-step cost is dominated by building the batch: about 0.23 MB per float32 sample, copied by numpy and sent to the GPU. That scales with samples, not with the number of steps, so half the steps × double the batch saves little. NoisyNet also draws about 1.6M Gaussian noise values per action selection (independent noise on a 3136×512 layer). Possible further speedups that don't change the learning math: store uint8 frames in the replay buffer (4× less copying), and factorized NoisyNet noise (Fortunato et al., 2018), which draws about 3.6k values instead of 1.6M.

Time budget at ~105–110 f/s: about 1.35–1.41M frames to stay ≤ 3.56 hr, i.e. 6–11% fewer frames than the baseline's 1.507M. Most of the baseline's time went to the final crawl from 14 to 19 (120 → 237 min), where its fixed ε = 0.02 random actions keep costing points. NoisyNet (shrinking noise) and Bootstrapped DQN (no random actions) can plausibly beat it there.

## Step 3 — PER on NoisyNet (`3a_noisynet_per_dqn.py`)

**Assignment goal (PDF step 3):** implement PER on the Step 2 algorithm, explain the sampling problem with baseline DQN and how PER fixes it, and **demonstrate a performance improvement**. No % target is given, and "performance" is not defined, so fewer frames to 19, less time to 19, or a higher score each count. State in the report which one improved.

**Setup:** an exact copy of `2a_noisynet_dqn.py` (the Step 2 winner: NoisyNet, random warm-up, train every 2, batch 64). The **only** change is that the uniform replay buffer becomes a proportional prioritized buffer (Schaul et al., 2016):
- priority = |TD error| + 1e-5; sampling probability ∝ priority^α with **α = 0.6**
- new transitions get the max priority seen so far, so each is trained on at least once soon
- importance-sampling weights w = (N·P(i))^-β / max w, with **β annealed 0.4 → 1.0 over 100k frames** (as in Lapan Ch. 7). Logged to TensorBoard as `beta`
- the loss is the IS-weighted squared TD error; after each step the sampled transitions get their new |TD error| as priority
- **sum tree** (vectorized numpy, O(log N)): measured at about 0.4 ms per step for sample(64) + update, including building the batch. The book's PER recomputed over the whole buffer and ran at ~71 f/s (37% fewer frames but 32% more time).

Verified locally: sampling frequencies match priority^α exactly (e.g. 0.201 expected vs 0.201 sampled), every tree node equals the sum of its children after ring-buffer wraparound, and IS weights max out at 1, with the highest-priority sample getting the smallest weight.

Command: `python 3a_noisynet_per_dqn.py --cuda --train-every 2 --batch-size 64`. Run tag `-noisy-per-te2-b64-warmup`.

**Comparison target = NoisyNet run 2:** 385,176 frames / 1.057 hr / 19.11 / ~101 f/s.

Interim (monitor log): 0.188 hr 68k -20.19 · 0.361 hr 121k -19.56 (snr 1.89/13.76) · 0.522 hr 171k -18.56 · **0.692 hr 222k -17.17** (110 games, last ~17 games ≈ -11) · 0.858 hr 274k -14.61 · **1.021 hr 324k -10.62** (139 games; last ~15 games ≈ +6, ~3,400 frames per game as rallies get longer). NoisyNet was at +9.30 at 322k. Steep climb started about 100k frames later than NoisyNet's; rough projection 500–550k frames / 1.6–1.8 hr. Speed **~83–88 f/s** (vs NoisyNet ~101): about 1.5 ms extra per frame on Colab's CPU for the tree ops and the per-step GPU→CPU copy of the TD errors. At the same point NoisyNet was much further along (-2.81 at 0.709 hr / 265k; recent games ≈ -3 at 205k).

**(Interim analysis at 222k frames) Can it beat NoisyNet's 385,176 frames? No (arithmetic).** At 222k frames the 100-game window summed about -1,717. Reaching a mean of 19 needs a sum of 1,900, i.e. at least ~94 more games **all won 21–0** (~1,750 frames each), so the earliest possible finish is about 387k frames, already past 385k. Combined with the lower f/s, this run cannot improve on NoisyNet in frames or time. It is left running to give its own frames/time-to-19 data point. Likely contributors: the small 10k buffer (the PER paper used 1M) and ordinary single-run variance. (Note: the IS weights scaling the loss down is NOT a strong explanation, because Adam is largely invariant to the overall gradient scale.)

**RESULT: SOLVED, but no improvement over NoisyNet.** Run name `Oct04_06-48-33_7688c770efb1-PongNoFrameskip-v4-noisy-per-te2-b64-warmup`. Log confirmation: `Solved in 580539 frames! (1.813 hr)`.

| Metric | Value | vs NoisyNet (Step 2): 385,176 frames / 1.057 hr / 19.11 / ~101 f/s |
|---|---|---|
| GPU | **Tesla T4** (nvidia-smi screenshot) | same GPU ✅ |
| Speed | ~89 f/s average (83–108) | about 12% slower per frame |
| Frames to converge | **580,539** (~250 games) | **+50.7% more frames** ❌ |
| Time to converge | **1.813 hr** | **+71.5% more time** ❌ |
| Final mean reward | 19.02 | about the same (stop bound) |
| vs baseline | 61.5% fewer frames, 54.2% less time | still well past the 10% target |

Later progress: 1.189 hr 376k -3.88 · 1.359 hr 430k +3.86 · 1.525 hr 483k 12.14 · 1.694 hr 537k 17.53 · **1.813 hr 581k 19.02**. snr_2 peaked at about 16.8 (vs about 17 for NoisyNet), so the noise was turned down by a similar amount.

**Report interpretation:** PER did **not** improve NoisyNet in this setup. Its steep climb started about 100–150k frames later than NoisyNet's, and its per-frame cost was about 12% higher. Possible reasons, which a single run can't separate: (1) NoisyNet alone was already very sample-efficient, leaving little room; (2) the 10k replay buffer is tiny for PER (the paper used 1M), so prioritization keeps re-drawing the same few transitions; (3) PER and NoisyNet both bias which experiences dominate learning, and they may interact; (4) single-run variance. On UCB, by contrast, PER gave 4.4% fewer frames (see below). Report both.

Screenshots: `results/plots/3a_noisynet_per/` (speed: full run; training_log_solved; nvidia_smi). The Colab TensorBoard view for beta/reward/reward_100 froze at 17k frames, so the full graphs are to be drawn from the event file with `tools/plot_runs.py`.
Artifacts: `results/{logs,tensorboard}/3a_noisynet_per*`.

## Step 3 (second variant) — PER on UCB Q-ensemble (`3b_ucb_per_dqn.py`)

**Why also UCB:** the PDF says "enhance the performance of the algorithm in step 2", and the lecture (Sept 14) says "the algorithm in step two, the one that you improved". UCB was also improved in Step 2 (2.75× faster than the baseline), so it qualifies. Reporting PER on **both** Step 2 algorithms is the most complete answer.

**Setup:** exact copy of `2b_ucb_ensemble_dqn.py` (5 heads, λ = 0.1, random warm-up, train every 2, batch 64). The only change is the same sum-tree PER buffer as 07 (α = 0.6, β 0.4 → 1 over 100k frames). Priority = |TD error| averaged over the K heads. The loss is the IS-weighted squared TD error summed over heads. Smoke-tested locally.

Command: `python 3b_ucb_per_dqn.py --cuda --heads 5 --train-every 2 --batch-size 64`. Run tag `-ucb-per-k5-l0.1-te2-b64-warmup`.

**Bar to beat = UCB (Step 2):** 568,001 frames / 1.438 hr / 19.03 / ~110 f/s.

**RESULT: SOLVED.** Run name `Oct04_07-46-06_a2e98865bbc9-PongNoFrameskip-v4-ucb-per-k5-l0.1-te2-b64-warmup`. Log confirmation: `Solved in 542750 frames! (1.517 hr)`.

| Metric | Value | vs UCB (Step 2): 568,001 frames / 1.438 hr / 19.03 / ~110 f/s |
|---|---|---|
| GPU | **Tesla T4** (nvidia-smi screenshot) | same GPU ✅ |
| Speed | ~99 f/s average (90–107) | about 10% slower per frame (the PER overhead) |
| Frames to converge | **542,750** (~245 games) | **4.4% fewer frames (-25,251)** ✅ |
| Time to converge | **1.517 hr (91.0 min)** | 5.5% more time (+4.7 min) |
| Final mean reward | 19.01 | about the same (both stop at the 19.0 bound) |
| vs baseline | 64.0% fewer frames, 61.7% less time | |

Progress (monitor log): 0.381 hr 136k -18.99 · 0.549 hr 193k -17.38 · 0.708 hr 250k -13.66 · 0.882 hr 315k -2.28 · 1.050 hr 377k 9.56 · 1.217 hr 436k 16.89 · 1.383 hr 495k **18.53** · **1.517 hr 543k 19.01**.

**Where PER helped (shape of the curve):** in the middle phase UCB+PER was slightly behind UCB at the same frames (≈ -2.3 at 315k vs UCB +1.6 at 294k; 9.6 at 377k vs 13.1 at 360k). It caught up by about 435k (16.89 vs 16.93 at 425k), and then the **final stretch was faster**: at about 495k frames PER was at 18.53 vs UCB 18.01, and it crossed 19 about 25k frames earlier. Its per-game reward graph is also visibly steadier after 450k (almost every game 19–21, few dips). Intuition for the report: once the agent wins most points, the few points it still loses are the high-TD-error ("surprising") transitions, and PER replays exactly those more often, so it fixes its last mistakes faster.

**Honest caveat:** a 4.4% frame reduction from a single run is within run-to-run variance (the author's same algorithm varied 87 vs 163 min). It is a measured improvement in sample efficiency, **not** in wall-clock time, because PER's per-frame overhead (~10%) outweighs the frame savings.

`beta` rose linearly from 0.4 to 1.0 over the first 100k frames, as designed (screenshot).
Screenshots: `results/plots/3b_ucb_per/` (beta, reward, reward_100, speed, training_log_solved, nvidia_smi).
Artifacts: `results/{logs,tensorboard}/3b_ucb_per*`.
