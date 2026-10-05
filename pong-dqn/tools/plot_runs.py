"""Redraw the report graphs in results/plots/ straight from the TensorBoard event files in
results/tensorboard/ (no TensorFlow needed). Usage, from the repo root: python tools/plot_runs.py
Needs matplotlib."""
import glob, os, struct, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


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
    """Event{wall_time=1, step=2, summary=5{value=1{tag=1, simple_value=2}}} -> {tag: [(step, value, wall)]}"""
    d = open(path, "rb").read(); i = 0; out = {}
    while i < len(d):
        n = struct.unpack("<Q", d[i:i + 8])[0]; rec = d[i + 12:i + 12 + n]; i += 12 + n + 4
        wall = step = None; summ = None
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


class MissingRun(Exception):
    pass


def events(folder):
    files = glob.glob(os.path.join(ROOT, folder, "**", "events.out.tfevents.*"), recursive=True)
    if not files:
        raise MissingRun(folder)
    return read_scalars(files[0])


def smooth(ys, w=0.6):  # TensorBoard-style EMA
    out, last = [], ys[0]
    for y in ys:
        last = last * w + (1 - w) * y; out.append(last)
    return out


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2); ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def fig():
    f, ax = plt.subplots(figsize=(10, 4.2), dpi=150); f.patch.set_facecolor(SURFACE); return f, ax


def kfmt(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x < 1e6 else f"{x/1e6:.2f}M"))


def save(f, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f.tight_layout(); f.savefig(path, facecolor=SURFACE); plt.close(f); print("wrote", os.path.relpath(path, ROOT))


# ---------- 1) Bootstrapped DQN tuned: full-run graphs ----------
def section_1():
    b = events("results/tensorboard/2x_bootstrap_tuned")
    dst = os.path.join(ROOT, "results/plots/2x_bootstrap_tuned")
    end = b["reward_100"][-1]; hrs_end = b["elapsed_hours"][-1][1]
    for tag, ylabel, title in [("reward", "Reward per game", "Bootstrapped DQN (tuned: K=5, train every 2, batch 64): reward per game"),
                               ("reward_100", "Mean reward, last 100 games", "Bootstrapped DQN (tuned): 100-game mean reward"),
                               ("speed", "Frames per second", "Bootstrapped DQN (tuned): training speed"),
                               ("elapsed_hours", "Elapsed hours", "Bootstrapped DQN (tuned): wall-clock time")]:
        xs = [p[0] for p in b[tag]]; ys = [p[1] for p in b[tag]]
        f, ax = fig()
        if tag == "reward":
            ax.plot(xs, ys, color=BLUE, alpha=0.25, linewidth=1)
            ax.plot(xs, smooth(ys), color=BLUE, linewidth=2)
        else:
            ax.plot(xs, ys, color=BLUE, linewidth=2)
        if tag in ("reward", "reward_100"):
            ax.axhline(19, color=INK2, linestyle="--", linewidth=1)
            ax.annotate("solve bound 19.0", (xs[0], 19), xytext=(4, 4), textcoords="offset points", color=INK2, fontsize=9)
        if tag == "reward_100":
            best = max(b[tag], key=lambda p: p[1])
            ax.plot([best[0]], [best[1]], "o", color=BLUE, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
            ax.annotate(f"peak {best[1]:.2f} @ {best[0]:,} frames", (best[0], best[1]), xytext=(0, -30), textcoords="offset points",
                        ha="center", color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
            ax.annotate(f"stopped: {end[1]:.2f} @ {end[0]:,} frames, {hrs_end:.2f} hr", (end[0], end[1]),
                        xytext=(0, -60), textcoords="offset points", ha="right", color=INK, fontsize=9,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        if tag == "speed":
            ax.set_ylim(0, 200)
        style(ax, title, "Frames", ylabel); kfmt(ax)
        save(f, os.path.join(dst, f"{tag}_full_3.01hr.png"))


# ---------- 2) Comparison: baseline vs NoisyNet vs Bootstrapped DQN (100-game mean vs time) ----------
def section_2():
    runs = [("Baseline (ε-greedy)", "results/tensorboard/1_baseline", BLUE, -22),
            ("NoisyNet", "results/tensorboard/2a_noisynet", ORANGE, 14),
            ("Bootstrapped DQN (tuned)", "results/tensorboard/2x_bootstrap_tuned", AQUA, -22),
            ("UCB Q-ensemble", "results/tensorboard/2b_ucb", YELLOW, 0)]
    os.makedirs(os.path.join(ROOT, "results/plots/comparison"), exist_ok=True)
    for xkey, xlabel, fname in [("hours", "Wall-clock hours (Tesla T4)", "reward_100_vs_time.png"),
                                 ("frames", "Frames", "reward_100_vs_frames.png")]:
        f, ax = fig()
        for name, folder, color, dy in runs:
            ev = events(folder)["reward_100"]
            t0 = ev[0][2]
            xs = [(w - t0) / 3600 for _, _, w in ev] if xkey == "hours" else [s for s, _, _ in ev]
            ys = [v for _, v, _ in ev]
            status = "solved" if ys[-1] > 19 else "stopped, not solved"
            when = f"{xs[-1]:.2f} hr" if xkey == "hours" else f"{xs[-1]:,.0f} frames"
            ax.plot(xs, ys, color=color, linewidth=2, label=f"{name}: {ys[-1]:.2f} at {when} ({status})")
            ax.plot([xs[-1]], [ys[-1]], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.axhline(19, color=INK2, linestyle="--", linewidth=1)
        ax.annotate("solve bound 19.0", (0, 19), xytext=(4, 4), textcoords="offset points", color=INK2, fontsize=9)
        if xkey == "hours":
            ax.axvline(3.56, color=INK2, linestyle=":", linewidth=1)
            ax.annotate("10% target ≤ 3.56 hr", (3.56, 0), xytext=(4, 0), textcoords="offset points", ha="left", color=INK2, fontsize=9)
        else:
            kfmt(ax)
        style(ax, "100-game mean reward: baseline vs Step 2 algorithms", xlabel, "Mean reward, last 100 games")
        leg = ax.legend(loc="lower right", frameon=True, fontsize=9, labelcolor=INK)
        leg.get_frame().set_facecolor(SURFACE); leg.get_frame().set_edgecolor(GRID)
        save(f, os.path.join(ROOT, "results/plots/comparison", fname))


# ---------- 3) NoisyNet + PER: full-run graphs (Colab TensorBoard froze at 17k frames) ----------
def section_3():
    npr = events("results/tensorboard/3a_noisynet_per")
    dst = os.path.join(ROOT, "results/plots/3a_noisynet_per")
    for tag, ylabel, title in [("reward", "Reward per game", "NoisyNet + PER: reward per game"),
                               ("reward_100", "Mean reward, last 100 games", "NoisyNet + PER: 100-game mean reward"),
                               ("snr_1", "Signal-to-noise ratio", "NoisyNet + PER: noise SNR, layer 1 (higher = less exploration)"),
                               ("snr_2", "Signal-to-noise ratio", "NoisyNet + PER: noise SNR, layer 2 (higher = less exploration)"),
                               ("beta", "β (importance-sampling strength)", "NoisyNet + PER: β annealed 0.4 → 1.0 over 100k frames"),
                               ("elapsed_hours", "Elapsed hours", "NoisyNet + PER: wall-clock time")]:
        xs = [p[0] for p in npr[tag]]; ys = [p[1] for p in npr[tag]]
        f, ax = fig()
        if tag == "reward":
            ax.plot(xs, ys, color=BLUE, alpha=0.25, linewidth=1); ax.plot(xs, smooth(ys), color=BLUE, linewidth=2)
        else:
            ax.plot(xs, ys, color=BLUE, linewidth=2)
        if tag in ("reward", "reward_100"):
            ax.axhline(19, color=INK2, linestyle="--", linewidth=1)
            ax.annotate("solve bound 19.0", (xs[0], 19), xytext=(4, 4), textcoords="offset points", color=INK2, fontsize=9)
        style(ax, title, "Frames", ylabel); kfmt(ax)
        save(f, os.path.join(dst, f"{tag}_full.png"))


# ---------- 4) Step 3 comparison: each Step 2 algorithm with vs without PER ----------
def section_4():
    for algo, plain, per, c1, c2, fname, outdir in [
            ("NoisyNet", "results/tensorboard/2a_noisynet", "results/tensorboard/3a_noisynet_per", ORANGE, BLUE, "step3_noisynet_vs_noisynet_per.png", "3a_noisynet_per"),
            ("UCB Q-ensemble", "results/tensorboard/2b_ucb", "results/tensorboard/3b_ucb_per", ORANGE, BLUE, "step3_ucb_vs_ucb_per.png", "3b_ucb_per")]:
        f, ax = fig()
        for name, folder, color in [(algo, plain, c1), (algo + " + PER", per, c2)]:
            ev = events(folder)["reward_100"]
            xs = [s for s, _, _ in ev]; ys = [v for _, v, _ in ev]
            hrs = (ev[-1][2] - ev[0][2]) / 3600
            ax.plot(xs, ys, color=color, linewidth=2, label=f"{name}: {ys[-1]:.2f} at {xs[-1]:,} frames, {hrs:.2f} hr")
            ax.plot([xs[-1]], [ys[-1]], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.axhline(19, color=INK2, linestyle="--", linewidth=1)
        ax.annotate("solve bound 19.0", (0, 19), xytext=(4, 4), textcoords="offset points", color=INK2, fontsize=9)
        style(ax, f"Step 3: {algo} with vs without PER (100-game mean)", "Frames", "Mean reward, last 100 games"); kfmt(ax)
        leg = ax.legend(loc="lower right", frameon=True, fontsize=9, labelcolor=INK)
        leg.get_frame().set_facecolor(SURFACE); leg.get_frame().set_edgecolor(GRID)
        save(f, os.path.join(ROOT, "results/plots", outdir, fname))


for section in (section_1, section_2, section_3, section_4):
    try:
        section()
    except MissingRun as e:
        print(f"skipped {section.__name__}: no TensorBoard data in {e} yet")
