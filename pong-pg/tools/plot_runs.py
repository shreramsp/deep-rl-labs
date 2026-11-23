"""Draw the Project 2 report graphs straight from the TensorBoard event files (no TensorFlow needed).

Usage, from pong-pg/:
    python tools/plot_runs.py [DATA_DIR] [OUT_DIR]
DATA_DIR holds one folder per run, named after its Colab notebook (Pong2_...), each with the run's
TensorBoard event file somewhere inside. The run's log is read (for the start-up minutes) from
training.log in that folder or from results/logs/<run>.log. Default: results/tensorboard.
OUT_DIR defaults to results/plots. The DQN baseline comes from ../pong-dqn/results/tensorboard/1_baseline.
Needs matplotlib.
"""
import glob
import math
import os
import re
import struct
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "tensorboard")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results", "plots")
DQN_BASELINE = os.path.join(HERE, "..", "pong-dqn", "results", "tensorboard", "1_baseline")

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#a3a29d"
# Categorical slots in fixed order (blue, orange, aqua, yellow, magenta, green, violet, red)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


# ---------- reading TensorBoard event files ----------
def varint(b, j):
    v = s = 0
    while True:
        x = b[j]; j += 1; v |= (x & 0x7F) << s; s += 7
        if x < 0x80:
            return v, j


def fields(b):
    j = 0
    while j < len(b):
        key, j = varint(b, j); f, t = key >> 3, key & 7
        if t == 0: v, j = varint(b, j)
        elif t == 1: v = b[j:j + 8]; j += 8
        elif t == 5: v = b[j:j + 4]; j += 4
        elif t == 2: n, j = varint(b, j); v = b[j:j + n]; j += n
        else: return
        yield f, t, v


def read_scalars(path):
    """{tag: [(frame, value, wall_time)]}"""
    d = open(path, "rb").read(); i = 0; out = {}
    while i < len(d):
        n = struct.unpack("<Q", d[i:i + 8])[0]; rec = d[i + 12:i + 12 + n]; i += 12 + n + 4
        wall = step = summ = None
        for f, t, v in fields(rec):
            if f == 1 and t == 1: wall = struct.unpack("<d", v)[0]
            elif f == 2 and t == 0: step = v
            elif f == 5 and t == 2: summ = v
        if summ is None: continue
        for f, t, val in fields(summ):
            if f != 1: continue
            tag = sv = None
            for f2, t2, v2 in fields(val):
                if f2 == 1: tag = v2.decode()
                elif f2 == 2 and t2 == 5: sv = struct.unpack("<f", v2)[0]
            if tag is not None and sv is not None:
                out.setdefault(tag, []).append((step, sv, wall))
    return out


class Run:
    def __init__(self, folder, label):
        path = folder if os.path.isabs(folder) else os.path.join(DATA, folder)
        files = sorted(glob.glob(os.path.join(path, "**", "events.out.tfevents.*"), recursive=True))
        if not files:
            raise FileNotFoundError(path)
        self.label, self.ev = label, read_scalars(files[0])
        # Hours since the script started: TensorBoard's first point is the first finished game,
        # so add the hours the log printed for that game (a minute or so).
        self.h0, self.final = 0.0, None
        for log in (os.path.join(path, "training.log"),
                    os.path.join(os.path.dirname(DATA), "logs", os.path.basename(path) + ".log")):
            if not os.path.exists(log): continue
            text = open(log).read()
            m = re.search(r"done 1 games.* ([\d.]+) h", text)
            if m: self.h0 = float(m.group(1))
            # The log's last game line is the final word (TensorBoard may lag it by a few games)
            last = re.findall(r"^(\d+): done \d+ games, .*mean reward ([-\d.]+), .* ([\d.]+) h$", text, re.M)
            if last: self.final = (int(last[-1][0]), float(last[-1][1]), float(last[-1][2]))
            break
        self.t0 = self.ev["reward_100"][0][2]

    def series(self, tag, x="frames"):
        pts = self.ev[tag]
        xs = [s for s, _, _ in pts] if x == "frames" else [self.h0 + (w - self.t0) / 3600 for _, _, w in pts]
        return xs, [v for _, v, _ in pts]

    def end(self):
        if self.final:
            return self.final
        (f, v, w) = self.ev["reward_100"][-1]
        return f, v, self.h0 + (w - self.t0) / 3600


# ---------- drawing ----------
def fig(h=4.6):
    f, ax = plt.subplots(figsize=(10, h), dpi=150); f.patch.set_facecolor(SURFACE); return f, ax


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2); ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def mfmt(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: "0" if x == 0 else f"{x/1e6:g}M"))


def legend(ax, loc="upper left"):
    if loc == "below":  # under the plot, so it never hides a line
        leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True, fontsize=8.5, labelcolor=INK)
    else:
        leg = ax.legend(loc=loc, frameon=True, fontsize=8.5, labelcolor=INK)
    leg.get_frame().set_facecolor(SURFACE); leg.get_frame().set_edgecolor(GRID)


def bound(ax):
    ax.axhline(19, color=INK2, linestyle="--", linewidth=1)
    ax.annotate("solved = 19.0", (ax.get_xlim()[1], 19), xytext=(-4, 4), textcoords="offset points",
                ha="right", color=INK2, fontsize=9)


def save(f, name):
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f.tight_layout(); f.savefig(path, facecolor=SURFACE); plt.close(f); print("wrote", path)


def outcome(r):
    f, v, h = r.end()
    return f"{v:.2f} at {f/1e6:.2f}M frames, {h:.2f} h" + ("" if v > 19 else " (not solved)")


def reward_plot(runs, x, title, name, solved_bound=True, ylim=None, loc="upper left"):
    f, ax = fig(6.2 if loc == "below" else 4.6)
    for r, color, dashed in runs:
        xs, ys = r.series("reward_100", x)
        ax.plot(xs, ys, color=color, linewidth=1.6 if dashed else 2, linestyle="--" if dashed else "-",
                label=f"{r.label}: {outcome(r)}")
        ax.plot([xs[-1]], [ys[-1]], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
    if ylim: ax.set_ylim(*ylim)
    if solved_bound: bound(ax)
    style(ax, title, "Hours of training (Colab T4)" if x == "hours" else "Frames", "Mean reward, last 100 games")
    if x == "frames": mfmt(ax)
    legend(ax, loc)
    save(f, name)


def main():
    # A2C: solved runs in color, the two settings that never learned in grey dashes
    a2c = [(Run("Pong2_a2c_untuned", "Default (lr 0.001, 4-step)"), C[0], False),
           (Run("Pong2_a2c_untuned_seed1", "Default, repeat (seed 1)"), C[4], False),
           (Run("Pong2_a2c_lr0.002", "lr 0.002"), C[1], False),
           (Run("Pong2_a2c_n8", "8-step returns"), C[2], False),
           (Run("Pong2_a2c_lr0.002_n8", "lr 0.002 + 8-step"), C[6], False),
           (Run("Pong2_a2c_lr0.003", "lr 0.003 (stopped)"), MUTED, True),
           (Run("Pong2_a2c_b32", "batch 32"), INK2, True)]
    for x in ("hours", "frames"):
        reward_plot(a2c, x, "A2C: every run (100-game mean reward)", f"a2c/a2c_reward_100_vs_{x}.png", loc="center right")

    # A2C entropy: how quickly each run stops playing randomly (ln 6 = 1.79 is random)
    f, ax = fig()
    for r, color, dashed in a2c:
        xs, ys = r.series("entropy", "frames")
        ax.plot(xs, ys, color=color, linewidth=1.4 if dashed else 1.8, linestyle="--" if dashed else "-", label=r.label)
    ax.axhline(math.log(6), color=INK2, linestyle=":", linewidth=1)
    style(ax, "A2C: policy entropy (1.79 = random over 6 buttons)", "Frames", "Entropy"); mfmt(ax)
    legend(ax, "lower left")
    save(f, "a2c/a2c_entropy.png")

    # REINFORCE
    rf = [(Run("Pong2_reinforce_untuned", "Plain (lr 1e-4)"), C[0], False),
          (Run("Pong2_reinforce_norm", "lr 1e-4 + return norm."), C[4], False),
          (Run("Pong2_reinforce_lr0.001", "lr 1e-3 (collapsed)"), C[7], False),
          (Run("Pong2_reinforce_lr0.001_ent0.01", "lr 1e-3 + entropy 0.01"), C[1], False),
          (Run("Pong2_reinforce_lr0.001_ent0.01_pt", "lr 1e-3 + entropy + per-point returns"), C[2], False),
          (Run("Pong2_reinforce_pt_norm", "lr 1e-3 + entropy + per-point + norm."), C[3], False),
          (Run("Pong2_reinforce_pt_norm_a2", "same, 2 buttons only (12 h)"), C[6], False)]
    for x in ("hours", "frames"):
        reward_plot(rf, x, "REINFORCE: every run (100-game mean reward)", f"reinforce/reinforce_reward_100_vs_{x}.png",
                    solved_bound=False, ylim=(-21.15, -19.6), loc="below")

    # REINFORCE entropy as a share of the random level (ln 6, or ln 2 for the 2-button run): 1.0 = random
    f, ax = fig(6.2)
    for r, color, _ in rf:
        xs, ys = r.series("entropy", "hours")
        top = math.log(2) if "2 buttons" in r.label else math.log(6)
        ax.plot(xs, [y / top for y in ys], color=color, linewidth=1.8, label=r.label)
    ax.axhline(1.0, color=INK2, linestyle=":", linewidth=1)
    ax.annotate("1.0 = fully random", (0, 1.0), xytext=(4, 4), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_ylim(0, 1.08)
    style(ax, "REINFORCE: entropy as a share of fully random (0 = always the same button)",
          "Hours of training (Colab T4)", "Entropy / random level")
    legend(ax, "below")
    save(f, "reinforce/reinforce_entropy.png")

    # DQN vs REINFORCE vs A2C
    dqn = Run(os.path.abspath(DQN_BASELINE), "DQN baseline (Project 1)")
    comp = [(dqn, C[0], False),
            (rf[0][0], C[7], False),
            (rf[5][0], C[3], False),
            (a2c[0][0], C[2], False),
            (a2c[2][0], C[1], False)]
    labels = ["DQN baseline (Project 1)", "REINFORCE, plain", "REINFORCE, best tuned (6 buttons)",
              "A2C, default", "A2C, tuned (lr 0.002)"]
    for (r, _, _), lab in zip(comp, labels):
        r.label = lab
    for x in ("hours", "frames"):
        reward_plot(comp, x, "DQN vs REINFORCE vs A2C (100-game mean reward)", f"comparison/dqn_reinforce_a2c_vs_{x}.png",
                    loc="center right")


if __name__ == "__main__":
    main()
