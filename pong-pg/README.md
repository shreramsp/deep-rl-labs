# CMPE 260 Project 2: Policy Gradients on Atari Pong

**Team:** Shreram Palanisamy, Sankalp Wahane

We train two policy-gradient methods on Pong and compare them with the DQN from [Project 1](../pong-dqn/):
1. **REINFORCE** (Williams, 1992) as the policy-gradient baseline, then tuned,
2. **A2C**, Advantage Actor-Critic (Mnih et al., 2016), tuned.

Settings are the same as Project 1: `PongNoFrameskip-v4`, the Project 1 wrappers (`lib/wrappers.py`), the same convolutional network body, γ = 0.99, Google Colab with a Tesla T4, and "solved" = the 100-game mean reward passes 19.0.

| File | What it is |
|---|---|
| `1_reinforce.py` | REINFORCE. Defaults are plain REINFORCE; tuning flags: `--lr`, `--games-per-update`, `--entropy-beta`, `--return-norm`, `--grad-clip`, `--point-returns`, `--two-actions` |
| `2_a2c.py` | A2C, ported from the book's Chapter 10 code with its default hyperparameters; tuning flags: `--lr`, `--envs`, `--batch-size`, `--reward-steps`, `--entropy-beta`, `--grad-clip` |
| `lib/wrappers.py` | Atari preprocessing, identical to Project 1 |
| `lib/pg_models.py` | Policy network (REINFORCE) and actor-critic network (A2C) |
| `colab/colab_commands.md` | Exact Colab cells for every run |

Both scripts take `--max-hours` to cap a run, save the best model as a `.dat` file, and log to TensorBoard (`runs/`).

Results will be added here as the runs finish.

## References
1. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8.
2. Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. *ICML*.
3. Lapan, M. *Deep Reinforcement Learning Hands-On*, Chapters 9–10. Packt.
4. Karpathy, A. (2016). Deep Reinforcement Learning: Pong from Pixels. Blog post, http://karpathy.github.io/2016/05/31/rl/ (per-point returns, `--point-returns`; UP/DOWN only, `--two-actions`).
