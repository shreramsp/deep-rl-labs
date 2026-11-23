# Project 2 Colab commands

Same workflow as Project 1: one Colab notebook per run, **Runtime → Change runtime type → T4 GPU**.
Every notebook clones the code from this public GitHub repo, so the code that was run is the code in the repo.

## All runs

All 14 runs are finished. This table is the record of exactly what ran: the notebook name, the command, and
the final result of each run. The results summary is in the [README](../README.md).

| Round | Notebook name | What it tests | Result | Training line (cell 2) |
|---|---|---|---|---|
| 1 | `Pong2_reinforce_untuned` | Plain REINFORCE, the required baseline | Done: −20.27 after 4 h (5,183,019 frames, 5,587 games), never learned | `!nohup python 1_reinforce.py --cuda --max-hours 4 > training.log 2>&1 &` |
| 1 | `Pong2_reinforce_norm` | Rescale the returns in each batch | Stopped at 3.51 h: −20.36 (4,971,022 frames), no learning | `!nohup python 1_reinforce.py --cuda --return-norm meanstd --max-hours 4 > training.log 2>&1 &` |
| 2 | `Pong2_a2c_untuned` | A2C with the default settings | **Solved: 19.06 in 7,958,200 frames, 3.48 h** | `!nohup python 2_a2c.py --cuda --max-hours 6 > training.log 2>&1 &` |
| 2 | `Pong2_a2c_lr0.003` | Learning rate 3x | Stopped at 2.77 h: −20.39 (6,279,114 frames), no learning | `!nohup python 2_a2c.py --cuda --lr 0.003 --max-hours 6 > training.log 2>&1 &` |
| 3 | `Pong2_a2c_b32` | Learn from 32 moves at a time instead of 128 | Done: −20.35 after 6 h (12,277,750 frames), never learned | `!nohup python 2_a2c.py --cuda --batch-size 32 --max-hours 6 > training.log 2>&1 &` |
| 3 | `Pong2_reinforce_lr0.001` | Learning rate 10x | Done: collapsed within 15 min (every game −21 from game ~250 on); −21.00 after 4 h | `!nohup python 1_reinforce.py --cuda --lr 0.001 --max-hours 4 > training.log 2>&1 &` |
| 4 | `Pong2_reinforce_lr0.001_ent0.01` | Fix the collapse: the same bigger steps plus an exploration bonus | Stopped at 2.33 h by a Colab disconnect: −20.27, no collapse, no learning | `!nohup python 1_reinforce.py --cuda --lr 0.001 --entropy-beta 0.01 --max-hours 4 > training.log 2>&1 &` |
| 4 | `Pong2_reinforce_lr0.001_ent0.01_pt` | The row above plus per-point returns: each move is judged only by the point it belongs to | Stopped at 2.27 h by a Colab disconnect: −20.31, no learning | `!nohup python 1_reinforce.py --cuda --lr 0.001 --entropy-beta 0.01 --point-returns --max-hours 4 > training.log 2>&1 &` |
| 5 | `Pong2_reinforce_pt_norm` | Best shot for REINFORCE: per-point returns + rescaled returns (a simple baseline) + exploration bonus, 6 h | Done: −20.33 after 6 h (8,718,841 frames, 9,392 games), no learning | `!nohup python 1_reinforce.py --cuda --lr 0.001 --entropy-beta 0.01 --point-returns --return-norm meanstd --max-hours 6 > training.log 2>&1 &` |
| 5 | `Pong2_a2c_lr0.002` | A learning rate between the one that worked (0.001) and the one that failed (0.003) | **Solved: 19.07 in 6,076,050 frames, 2.67 h** | `!nohup python 2_a2c.py --cuda --lr 0.002 --max-hours 6 > training.log 2>&1 &` |
| 6 | `Pong2_a2c_n8` | 8-step returns instead of 4, so real points reach the learning signal sooner | **Solved: 19.02 in 6,133,200 frames, 2.63 h** | `!nohup python 2_a2c.py --cuda --reward-steps 8 --max-hours 6 > training.log 2>&1 &` |
| 6 | `Pong2_reinforce_pt_norm_a2` | Same as `Pong2_reinforce_pt_norm`, but only 2 buttons (UP/DOWN, as in Karpathy 2016) instead of 6; overnight, 12 h. Start after `Pong2_reinforce_pt_norm` ends | Done: −20.36 after 12 h (16,641,505 frames, 16,306 games), no learning (entropy stayed at the random level, 0.693) | `!nohup python 1_reinforce.py --cuda --lr 0.001 --entropy-beta 0.01 --point-returns --return-norm meanstd --two-actions --max-hours 12 > training.log 2>&1 &` |
| 7 | `Pong2_a2c_lr0.002_n8` | Both A2C winners together (lr 0.002 and 8-step returns): do the two gains stack? | **Solved: 19.01 in 5,875,200 frames, 2.84 h** (fewest frames, but not faster: the gains don't stack) | `!nohup python 2_a2c.py --cuda --lr 0.002 --reward-steps 8 --max-hours 6 > training.log 2>&1 &` |
| 8 | `Pong2_a2c_untuned_seed1` | Repeat of the default A2C with a fixed seed: how much do results vary by luck alone? | **Solved: 19.04 in 7,508,200 frames, 3.24 h** (luck alone: ~6% from the first default run) | `!nohup python 2_a2c.py --cuda --seed 1 --max-hours 6 > training.log 2>&1 &` |

Each run stops by itself when the 100-game mean passes 19.0 ("Solved in N frames!"),
or when it hits the time cap ("Stopped: time limit ...").

## Cell 1: setup (same in every notebook)
```python
!git clone https://github.com/shreramsp/deep-rl-labs.git
%cd deep-rl-labs/pong-pg
!git log -1 --format="%h %s"                 # records which commit was run
!pip install -q "gymnasium[atari]" ale-py opencv-python-headless tensorboardX
!nvidia-smi --query-gpu=name --format=csv    # should say Tesla T4
```

## Cell 2: start training
Paste the one training line for this notebook from the table above. Then check it started:
```python
!sleep 180; head -n 1 training.log; tail -n 3 training.log
```
It waits 3 minutes because A2C plays 50 games at once, so its first score line appears only after 2-3 minutes.

## Cell 3: TensorBoard
```python
%load_ext tensorboard
%tensorboard --logdir runs
```
Graphs to watch: `reward_100` (the 100-game mean), `entropy` (1.79 = still pressing buttons at random), `speed`.

## Cell 4: keep-alive monitor (run last and leave it running)
Prints the latest training line every 2 minutes so Colab always sees an active, printing cell,
and copies the log, TensorBoard files and best model to Google Drive every 10 minutes.
It only stops when the run finishes ("Solved" or "Stopped") or the training has crashed.
**Set `RUN` to this notebook's name**, or two notebooks will back up into the same Drive folder.
```python
RUN = "Pong2_reinforce_untuned"          # MUST match this notebook's name

import time, subprocess, os
from google.colab import drive
drive.mount('/content/drive')
BACKUP = "/content/drive/MyDrive/" + RUN
os.makedirs(BACKUP, exist_ok=True)
i = 0
while True:
    tail = subprocess.run(["tail", "-n", "1", "training.log"], capture_output=True, text=True).stdout.strip()
    print(time.strftime("%H:%M"), "|", tail, flush=True)
    if i % 5 == 0:                                   # every 10 minutes
        os.system(f"cp -r runs *.dat training.log '{BACKUP}/' 2>/dev/null")
    if "Solved" in tail or "Stopped" in tail:
        os.system(f"cp -r runs *.dat training.log '{BACKUP}/' 2>/dev/null")
        print("DONE. Results are in Google Drive:", BACKUP)
        break
    if "Error" in tail or time.time() - os.path.getmtime("training.log") > 1200:
        print("WARNING: training has stopped (crash, or no new log line for 20 minutes). Check training.log")
        break
    i += 1
    time.sleep(120)
```
The training prints a line after every game, so a log that stops growing means the training stopped.

## Keep the Mac awake
Colab runs in the browser tab, so the Mac must not sleep. In Terminal, before leaving it:
```bash
caffeinate -dis -t 25200      # keeps the screen and CPU awake for 7 hours; Ctrl+C to stop
```
Keep the Mac plugged in and the Colab tabs open.

## Download results at the end
```python
!zip -r results.zip runs *.dat training.log
from google.colab import files
files.download("results.zip")
```

## TensorBoard comparison notebook (`Pong2_tensorboard_compare`)

No GPU needed (Runtime → Change runtime type → CPU). It loads every run's TensorBoard file from this
repo (`results/tensorboard/`) and shows three TensorBoards with short run names.

**Cell 1: get the repo**
```python
!git clone -q --depth 1 https://github.com/shreramsp/deep-rl-labs.git
%cd /content/deep-rl-labs/pong-pg
```

**Cell 2: group the runs into three views**
```python
import glob, os
TB = "/content/deep-rl-labs/pong-pg/results/tensorboard"
DQN = "/content/deep-rl-labs/pong-dqn/results/tensorboard/1_baseline"
views = {
    "tb_a2c": {
        "a2c_default": "Pong2_a2c_untuned",
        "a2c_default_repeat": "Pong2_a2c_untuned_seed1",
        "a2c_lr0.002": "Pong2_a2c_lr0.002",
        "a2c_8step": "Pong2_a2c_n8",
        "a2c_lr0.002_8step": "Pong2_a2c_lr0.002_n8",
        "a2c_lr0.003": "Pong2_a2c_lr0.003",
        "a2c_batch32": "Pong2_a2c_b32",
    },
    "tb_reinforce": {
        "reinforce_plain": "Pong2_reinforce_untuned",
        "reinforce_norm": "Pong2_reinforce_norm",
        "reinforce_lr0.001": "Pong2_reinforce_lr0.001",
        "reinforce_lr0.001_ent": "Pong2_reinforce_lr0.001_ent0.01",
        "reinforce_lr0.001_ent_pt": "Pong2_reinforce_lr0.001_ent0.01_pt",
        "reinforce_pt_norm": "Pong2_reinforce_pt_norm",
        "reinforce_pt_norm_2buttons": "Pong2_reinforce_pt_norm_a2",
    },
    "tb_compare": {
        "1_dqn_baseline": DQN,
        "2_reinforce_plain": "Pong2_reinforce_untuned",
        "3_reinforce_best_tuned": "Pong2_reinforce_pt_norm",
        "4_a2c_default": "Pong2_a2c_untuned",
        "5_a2c_tuned_lr0.002": "Pong2_a2c_lr0.002",
    },
}
for view, runs in views.items():
    for name, src in runs.items():
        files = glob.glob(os.path.join(TB, src, "**", "events.out.tfevents.*"), recursive=True)
        assert files, "missing: " + src
        os.makedirs(f"/content/{view}/{name}", exist_ok=True)
        for f in files:
            link = f"/content/{view}/{name}/" + os.path.basename(f)
            if not os.path.exists(link):
                os.symlink(f, link)
    print(view, "->", len(runs), "runs")
%load_ext tensorboard
```

**Cell 3: all A2C runs**
```python
%tensorboard --logdir /content/tb_a2c --port 6006
```

**Cell 4: all REINFORCE runs**
```python
%tensorboard --logdir /content/tb_reinforce --port 6007
```

**Cell 5: DQN vs REINFORCE vs A2C**
```python
%tensorboard --logdir /content/tb_compare --port 6008
```

For the screenshots, in each TensorBoard:
- Type `reward_100` in the tag filter box (the 100-game mean). `entropy` is also worth a screenshot for A2C and REINFORCE.
- Click the gear icon (top right) and set **Smoothing** to 0, because `reward_100` is already an average.
- The x-axis shows **Step** by default, which here means frames. In the same settings panel, set the horizontal axis to **Relative** to show hours of training instead.
- In Cell 5, DQN logged fewer graphs than the policy-gradient runs, so only `reward_100`, `reward` and `speed` show all five runs.
