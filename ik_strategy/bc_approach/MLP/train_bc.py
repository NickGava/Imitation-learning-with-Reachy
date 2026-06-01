'''
MLP/train_bc.py
=============================================================================
Trains the Behavioral Cloning MLP for a single exercise using k-fold
cross validation. Produces K independent models, one per fold.

Architecture : 32 -> [256, ReLU] -> [256, ReLU] -> 16
Input        : [q(t), dq(t)]  - current position + velocity (32 values)
Output       : Δq(t)          - joint delta (16 values)
Loss         : MSE
Optimizer    : Adam (lr=1e-3)
Early stopping: patience=20 on validation loss
Noise aug.   : Gaussian noise on state input during training (std=0.1)

K-fold: videos are divided into K folds using a composite subject_video key.
K is computed as ceil(N_videos * K_FOLDS_RATIO).

--- Input ---
  data/dataset/exercise_XXX/bc_dataset.csv

--- Output (single run) ---
  data/dataset/exercise_XXX/n_XX/MLP/bc_model_fold_0.pth  ...
  data/dataset/exercise_XXX/n_XX/MLP/scaler_fold_0.pkl    ...
  data/dataset/exercise_XXX/n_XX/MLP/loss_curve.png

--- Output (multi-run, --run N) ---
  data/dataset/exercise_XXX/n_XX/MLP/run_N/bc_model_fold_0.pth  ...
  data/dataset/exercise_XXX/n_XX/MLP/run_N/scaler_fold_0.pkl    ...
  data/dataset/exercise_XXX/n_XX/MLP/run_N/loss_curve.png

Usage:
  py -m bc_approach.MLP.train_bc 1
  py -m bc_approach.MLP.train_bc 1 --run 2   # training run 2 (different seed)
'''

import argparse
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from utilities.config import DATA_ROOT, JOINT_COLS
from utilities.split_utils import split_name

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATE_COLS  = [f'q_{j}'   for j in JOINT_COLS]
VEL_COLS    = [f'dq_{j}'  for j in JOINT_COLS]
ACTION_COLS = [f'act_{j}' for j in JOINT_COLS]
INPUT_COLS  = STATE_COLS + VEL_COLS
OUTPUT_COLS = ACTION_COLS

K_FOLDS_RATIO = 0.25
HIDDEN_SIZE   = 256
BATCH_SIZE    = 64
LR            = 1e-3
PATIENCE      = 20
MAX_EPOCHS    = 500
RANDOM_SEED   = 42
NOISE_STD     = 0.1


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BCPolicyMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size,  hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------
def _train_fold(X_trn, y_trn, X_val, y_val, device, fold_idx, output_dir):
    '''
    Train a single MLP fold on pre-split train/val arrays.

    Fits a StandardScaler on training data, trains the MLP with early
    stopping (patience=PATIENCE), and saves the best checkpoint and
    scaler to output_dir.

    Returns (best_val_loss, train_losses, val_losses).
    '''
    scaler  = StandardScaler()
    X_trn_n = scaler.fit_transform(X_trn)
    X_val_n = scaler.transform(X_val)

    scaler_path = output_dir / f'scaler_fold_{fold_idx}.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    X_trn_t = torch.tensor(X_trn_n, dtype=torch.float32).to(device)
    y_trn_t = torch.tensor(y_trn,   dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_n, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val,   dtype=torch.float32).to(device)

    trn_loader = DataLoader(TensorDataset(X_trn_t, y_trn_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

    model      = BCPolicyMLP(len(INPUT_COLS), HIDDEN_SIZE, len(OUTPUT_COLS)).to(device)
    criterion  = nn.MSELoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    model_path = output_dir / f'bc_model_fold_{fold_idx}.pth'

    best_val_loss  = float('inf')
    best_epoch     = 0
    patience_count = 0
    train_losses, val_losses = [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        trn_loss = 0.0
        for Xb, yb in trn_loader:
            optimizer.zero_grad()
            if NOISE_STD > 0:
                Xb = Xb + torch.randn_like(Xb) * NOISE_STD
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            trn_loss += loss.item() * len(Xb)
        trn_loss /= len(X_trn)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                val_loss += criterion(model(Xb), yb).item() * len(Xb)
        val_loss /= len(X_val)

        train_losses.append(trn_loss)
        val_losses.append(val_loss)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss  = val_loss
            best_epoch     = epoch
            patience_count = 0
            torch.save({
                'model_state': model.state_dict(),
                'state_dim':   len(INPUT_COLS),
                'action_dim':  len(OUTPUT_COLS),
                'hidden_size': HIDDEN_SIZE,
                'n_layers':    2,
                'val_loss':    val_loss,
                'hparams': {
                    'hidden_size': HIDDEN_SIZE,
                    'lr':          LR,
                    'batch_size':  BATCH_SIZE,
                    'noise_std':   NOISE_STD,
                }
            }, model_path)
        else:
            patience_count += 1

        if epoch % 10 == 0 or is_best:
            marker = ' ←' if is_best else ''
            print(f'  {epoch:>5}  {trn_loss:>12.6f}  {val_loss:>12.6f}{marker}')

        if patience_count >= PATIENCE:
            print(f'  Early stopping at epoch {epoch}.')
            break

    print(f'  Best: epoch {best_epoch}, val loss = {best_val_loss:.6f}')
    return best_val_loss, train_losses, val_losses


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------
def _save_loss_curve(all_train, all_val, path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(all_train)))
    for i, (tl, vl) in enumerate(zip(all_train, all_val)):
        ax.plot(tl, color=colors[i], lw=1.2, alpha=0.7, label=f'fold {i} train')
        ax.plot(vl, color=colors[i], lw=1.2, alpha=0.7, linestyle='--', label=f'fold {i} val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE loss')
    ax.set_title('MLP - K-fold training curves')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Loss curve saved → {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train BC MLP (k-fold) for one exercise.')
    parser.add_argument('exercise', type=int, help='Exercise number (e.g. 1)')
    parser.add_argument('--n-demos', type=int, default=55, choices=[10,25,55])
    parser.add_argument('--run', type=int, default=None, help='Training run index (1-based). Saves to MLP/run_N/ with a different random seed for reproducible independence.')
    args = parser.parse_args()

    split_dir    = DATA_ROOT / 'dataset' / f'exercise_{args.exercise:03d}' / split_name(args.n_demos)
    dataset_path = split_dir / 'bc_dataset.csv'

    # ── Run-aware output directory and seed ──────────────────────────────────
    if args.run is not None:
        output_dir    = split_dir / 'MLP' / f'run_{args.run}'
        effective_seed = RANDOM_SEED + (args.run - 1) * 137
        print(f'Training run {args.run}  (seed={effective_seed})')
    else:
        output_dir    = split_dir / 'MLP'
        effective_seed = RANDOM_SEED
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading {dataset_path} ...')
    if not dataset_path.exists():
        print(f'Error: dataset not found → {dataset_path}')
        return

    df = pd.read_csv(dataset_path)
    print(f'  {len(df)} samples\n')

    df['video_id'] = df['subject'].astype(str) + '_' + df['video'].astype(str)

    X = df[INPUT_COLS].values.astype(np.float32)
    y = df[OUTPUT_COLS].values.astype(np.float32)

    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)

    video_ids = df['video_id'].unique()
    np.random.shuffle(video_ids)
    K_FOLDS = max(2, math.ceil(len(video_ids) * K_FOLDS_RATIO))
    folds   = np.array_split(video_ids, K_FOLDS)

    print(f'K-fold cross validation  (K={K_FOLDS}, ratio={K_FOLDS_RATIO})')
    print(f'  Total videos : {len(video_ids)}')
    for i, fold in enumerate(folds):
        print(f'  Fold {i}       : val videos {sorted(fold.tolist())}')
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Model: {len(INPUT_COLS)} → [{HIDDEN_SIZE}, ReLU] → [{HIDDEN_SIZE}, ReLU] → {len(OUTPUT_COLS)}')
    print(f'Parameters: {sum(p.numel() for p in BCPolicyMLP(len(INPUT_COLS), HIDDEN_SIZE, len(OUTPUT_COLS)).parameters()):,}\n')

    fold_val_losses = []
    all_train_curves, all_val_curves = [], []

    for fold_idx in range(K_FOLDS):
        val_ids = set(folds[fold_idx].tolist())
        trn_ids = set(video_ids.tolist()) - val_ids

        trn_mask = df['video_id'].isin(trn_ids).values
        val_mask = df['video_id'].isin(val_ids).values
        X_trn, y_trn = X[trn_mask], y[trn_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        print(f'{"="*50}')
        print(f'Fold {fold_idx}  |  train: {len(X_trn)} samples  val: {len(X_val)} samples')
        print(f'{"="*50}')
        print(f"{'Epoch':>6}  {'Train loss':>12}  {'Val loss':>12}")
        print('-' * 36)

        best_val, tl, vl = _train_fold(
            X_trn, y_trn, X_val, y_val, device, fold_idx, output_dir)

        fold_val_losses.append(best_val)
        all_train_curves.append(tl)
        all_val_curves.append(vl)
        print()

    val_arr = np.array(fold_val_losses)
    print(f'\n{"="*50}')
    print(f'K-fold summary  [MLP]')
    print(f'{"="*50}')
    for i, v in enumerate(fold_val_losses):
        print(f'  Fold {i} val loss : {v:.6f}')
    print(f'  Mean ± std      : {val_arr.mean():.6f} ± {val_arr.std():.6f}')
    print(f'  Best fold       : {int(np.argmin(val_arr))}')
    print(f'{"="*50}\n')

    _save_loss_curve(all_train_curves, all_val_curves, output_dir / 'loss_curve.png')


if __name__ == '__main__':
    main()
