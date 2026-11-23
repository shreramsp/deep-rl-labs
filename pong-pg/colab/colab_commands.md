# Project 2 Colab commands

Same workflow as Project 1: one Colab notebook per run, **Runtime → Change runtime type → T4 GPU**.
Every notebook clones the code from this public GitHub repo, so the code that was run is the code in the repo.

## All runs

Two Colab sessions at a time. Start the **Next** rows first; each round pairs runs with the same time cap,
so both sessions free up together.

| Round | Notebook name | What it tests | Status | Training line (cell 2) |
|---|---|---|---|---|
| 1 | `Pong2_reinforce_untuned` | Plain REINFORCE, the required baseline | Done: -20.31 after 4 h, never learned | `!nohup python 1_reinforce.py --cuda --max-hours 4 > training.log 2>&1 &` |
| 1 | `Pong2_reinforce_norm` | Rescale the returns in each batch | Stopped at 2.1 h: -20.46, no learning | `!nohup python 1_reinforce.py --cuda --return-norm meanstd --max-hours 4 > training.log 2>&1 &` |
| 2 | `Pong2_a2c_untuned` | A2C with the default settings | **Solved: 19.06 in 7,958,200 frames, 3.48 h** | `!nohup python 2_a2c.py --cuda --max-hours 6 > training.log 2>&1 &` |
| 2 | `Pong2_a2c_lr0.003` | Learning rate 3x | Stopped at 2.6 h: -20.3, no learning | `!nohup python 2_a2c.py --cuda --lr 0.003 --max-hours 6 > training.log 2>&1 &` |
| 3 | `Pong2_a2c_b32` | Learn from 32 moves at a time instead of 128 | Done: -20.40 after 6 h, never learned | `!nohup python 2_a2c.py --cuda --batch-size 32 --max-hours 6 > training.log 2>&1 &` |
| 3 | `Pong2_reinforce_lr0.001` | Learning rate 10x | Done: collapsed, every game -21 | `!nohup python 1_reinforce.py --cuda --lr 0.001 --max-hours 4 > training.log 2>&1 &` |
| 4 | `Pong2_reinforce_lr0.001_ent0.01` | Fix the collapse: the same bigger steps plus an exploration bonus | Stopped at ~2.1 h by a Colab disconnect: -20.29, no collapse, no learning | `!nohup python 1_reinforce.py --cuda --lr 0.001 --entropy-beta 0.01 --max-hours 4 > training.log 2>&1 &` |
| 4 | `Pong2_reinforce_lr0.001_ent0.01_pt` | The row above plus per-point returns: each move is judged only by the point it belongs to | Stopped at ~2.1 h by a Colab disconnect: -20.34, no learning | `!nohup python 1_reinforce.py --cuda --lr 0.001 --entropy-beta 0.01 --point-returns --max-hours 4 > training.log 2>&1 &` |
| 5 | `Pong2_reinforce_pt_norm` | Best shot for REINFORCE: per-point returns + rescaled returns (a simple baseline) + exploration bonus, 6 h | Done: −20.27 after 6 h (8,718,841 frames, ~9,400 games), no learning | `!nohup python 1_reinforce.py --cuda --lr 0.001 --entropy-beta 0.01 --point-returns --return-norm meanstd --max-hours 6 > training.log 2>&1 &` |
| 5 | `Pong2_a2c_lr0.002` | A learning rate between the one that worked (0.001) and the one that failed (0.003) | **Solved: 19.07 in 6,076,050 frames, ~2.7 h (best A2C so far)** | `!nohup python 2_a2c.py --cuda --lr 0.002 --max-hours 6 > training.log 2>&1 &` |
| 6 | `Pong2_a2c_n8` | 8-step returns instead of 4, so real points reach the learning signal sooner | **Solved: 19.02 in 6,133,200 frames, ~2.64 h** | `!nohup python 2_a2c.py --cuda --reward-steps 8 --max-hours 6 > training.log 2>&1 &` |
| 6 | `Pong2_reinforce_pt_norm_a2` | Same as `Pong2_reinforce_pt_norm`, but only 2 buttons (UP/DOWN, as in Karpathy 2016) instead of 6; overnight, 12 h. Start after `Pong2_reinforce_pt_norm` ends | Running (−20.34 at 3.0 h, no learning yet) | `!nohup python 1_reinforce.py --cuda --lr 0.001 --entropy-beta 0.01 --point-returns --return-norm meanstd --two-actions --max-hours 12 > training.log 2>&1 &` |
| 7 | `Pong2_a2c_lr0.002_n8` | Both A2C winners together (lr 0.002 and 8-step returns): do the two gains stack? | **Solved: 19.01 in 5,875,200 frames, ~2.85 h** (fewest frames, but not faster: the gains don't stack) | `!nohup python 2_a2c.py --cuda --lr 0.002 --reward-steps 8 --max-hours 6 > training.log 2>&1 &` |
| 8 | `Pong2_a2c_untuned_seed1` | Repeat of the default A2C with a fixed seed: how much do results vary by luck alone? | **Next** (overnight) | `!nohup python 2_a2c.py --cuda --seed 1 --max-hours 6 > training.log 2>&1 &` |

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
