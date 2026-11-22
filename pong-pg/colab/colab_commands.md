# Project 2 Colab commands

Same workflow as Project 1: one Colab notebook per run, **Runtime → Change runtime type → T4 GPU**.
Every notebook clones the code from this public GitHub repo, so the code that was run is the code in the repo.

## First batch: 4 notebooks, started together

| Notebook name | What it tests | Training line (cell 2) |
|---|---|---|
| `Pong2_reinforce_untuned` | Plain REINFORCE, the required baseline | `!nohup python 1_reinforce.py --cuda --max-hours 4 > training.log 2>&1 &` |
| `Pong2_reinforce_norm` | Tuning 1: scale the returns in each batch | `!nohup python 1_reinforce.py --cuda --return-norm meanstd --max-hours 4 > training.log 2>&1 &` |
| `Pong2_a2c_untuned` | A2C with the book's settings | `!nohup python 2_a2c.py --cuda --max-hours 6 > training.log 2>&1 &` |
| `Pong2_a2c_lr0.003` | Tuning 1: the author's best learning rate | `!nohup python 2_a2c.py --cuda --lr 0.003 --max-hours 6 > training.log 2>&1 &` |

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
!sleep 60; head -n 1 training.log; tail -n 3 training.log
```

## Cell 3: TensorBoard
```python
%load_ext tensorboard
%tensorboard --logdir runs
```
Graphs to watch: `reward_100` (the 100-game mean), `entropy` (1.79 = still pressing buttons at random), `speed`.

## Cell 4: keep-alive monitor (run last and leave it running)
Prints progress every 5 minutes so Colab sees an active cell and does not disconnect,
and copies the log, TensorBoard files and best model to Google Drive each time,
so nothing is lost if the session dies anyway. It stops when the run finishes.
```python
RUN = "Pong2_reinforce_untuned"          # change to this notebook's name

import time, subprocess, os
from google.colab import drive
drive.mount('/content/drive')
BACKUP = "/content/drive/MyDrive/" + RUN
os.makedirs(BACKUP, exist_ok=True)
start = time.time()
while True:
    tail = subprocess.run(["tail", "-n", "2", "training.log"], capture_output=True, text=True).stdout
    print(time.strftime("%H:%M"), "| %.2f h elapsed |" % ((time.time() - start) / 3600), tail.strip(), flush=True)
    os.system(f"cp -r runs *.dat training.log '{BACKUP}/' 2>/dev/null")
    if "Solved" in tail or "Stopped" in tail:
        print("DONE. Results are in Google Drive:", BACKUP)
        break
    if subprocess.run(["pgrep", "-f", "python [12]_"], capture_output=True).returncode != 0:
        print("WARNING: training is not running anymore. Check training.log")
        break
    time.sleep(300)
```

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
