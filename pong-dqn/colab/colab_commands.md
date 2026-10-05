# Colab commands used for every run

All runs used Google Colab with **Runtime → Change runtime type → T4 GPU**, one notebook per run.

## 1. Setup
```python
!git clone https://github.com/shreramsp/deep-rl-labs.git
%cd deep-rl-labs/pong-dqn
!pip install -q gymnasium[atari] ale-py opencv-python-headless tensorboardX
!nvidia-smi        # record the GPU (should say Tesla T4)
```

## 2. Start training (one line per notebook)
```python
!nohup python 1_baseline_dqn.py --cuda > training_baseline.log 2>&1 &
!nohup python 2a_noisynet_dqn.py --cuda --train-every 2 --batch-size 64 > training_noisynet.log 2>&1 &
!nohup python 2b_ucb_ensemble_dqn.py --cuda --heads 5 --train-every 2 --batch-size 64 > training_ucb.log 2>&1 &
!nohup python 3a_noisynet_per_dqn.py --cuda --train-every 2 --batch-size 64 > training_noisynet_per.log 2>&1 &
!nohup python 3b_ucb_per_dqn.py --cuda --heads 5 --train-every 2 --batch-size 64 > training_ucb_per.log 2>&1 &
!nohup python experiments/2x_bootstrapped_dqn.py --cuda --heads 5 --train-every 2 --batch-size 64 > training_bootstrap.log 2>&1 &
```
Check that it started: `!head -n 3 <log>` prints the `Config:` line for scripts 2a–3b.

## 3. TensorBoard
```python
%load_ext tensorboard
%tensorboard --logdir runs
```

## 4. Monitor (keeps the session active, backs up to Google Drive, stops when solved)
```python
SCRIPT = "2a_noisynet_dqn.py"          # change per notebook
LOG    = "training_noisynet.log"       # change per notebook

import time, subprocess, os
from google.colab import drive
drive.mount('/content/drive')
BACKUP = "/content/drive/MyDrive/pong_" + SCRIPT.replace(".py", "").replace("/", "_")
os.makedirs(BACKUP, exist_ok=True)

while True:
    tail = subprocess.run(["tail", "-n", "2", LOG], capture_output=True, text=True).stdout
    print(time.strftime("%H:%M"), tail, flush=True)
    os.system(f"cp -r runs *.dat {LOG} '{BACKUP}/' 2>/dev/null")
    if "Solved" in tail:
        print("DONE — results are in Google Drive:", BACKUP)
        break
    if subprocess.run(["pgrep", "-f", SCRIPT], capture_output=True).returncode != 0:
        print("WARNING: training process is not running anymore — check the log")
        break
    time.sleep(600)
```

## 5. Download results
```python
!zip -r results.zip runs *.dat training_*.log
from google.colab import files
files.download("results.zip")
```

> The runs in `results/` were made before the scripts were renamed to their step-based names (e.g. `2a_noisynet_dqn.py` was `04_dqn_noisy_pong.py`). The code is otherwise identical, and the TensorBoard run names inside `results/tensorboard/` keep their original tags (`-noisy-te2-b64-warmup`, `-ucb-k5-l0.1-te2-b64-warmup`, …).
