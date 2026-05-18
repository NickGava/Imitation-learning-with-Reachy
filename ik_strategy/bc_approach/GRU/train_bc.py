'''
GRU/train_bc.py
=============================================================================
Trains the Behavioral Cloning GRU for a single exercise using k-fold
cross validation. Produces K independent models, one per fold.

Architecture : GRU(input=32, hidden=256) -> Linear(256 -> 16)
Input        : sequence of [q, dq] values, shape (SEQ_LEN, 32)
Output       : Δq(t)  — joint delta (16 values)
Loss         : MSE
Optimizer    : Adam (lr=1e-3)
Early stopping: patience=20 on validation loss
Noise aug.   : Gaussian noise on input sequences during training (std=0.1)

K-fold: videos are divided into K folds using a composite subject_video key
to correctly identify unique videos across multiple subjects.
K is computed as ceil(N_videos * K_FOLDS_RATIO).

--- Input ---
  data/dataset/exercise_XXX/bc_dataset.csv

--- Output ---
  data/dataset/exercise_XXX/GRU/bc_model_fold_0.pth  ...  bc_model_fold_K.pth
  data/dataset/exercise_XXX/GRU/scaler_fold_0.pkl    ...  scaler_fold_K.pkl
  data/dataset/exercise_XXX/GRU/loss_curve.png

Usage:
  py -m bc_approach.GRU.train_bc 1
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
N_JOINTS    = len(JOINT_COLS)
STATE_COLS  = [f'q_{j}'   for j in JOINT_COLS]
VEL_COLS    = [f'dq_{j}'  for j in JOINT_COLS]
ACTION_COLS = [f'act_{j}' for j in JOINT_COLS]

N_INPUT       = N_JOINTS * 2  # [q, dq] per timestep

SEQ_LEN       = 20
K_FOLDS_RATIO = 0.25
HIDDEN_SIZE   = 256
N_LAYERS      = 1
BATCH_SIZE    = 64
LR            = 1e-3
PATIENCE      = 20
MAX_EPOCHS    = 500
RANDOM_SEED   = 42
NOISE_STD     = 0.1


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BCPolicyGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 1):
        super().__init__()
        self.gru  = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------
def build_sequences(df: pd.DataFrame, seq_len: int):
    """
    Creates sliding-window sequences from bc_dataset.csv.
    Windows never cross video boundaries (identified by video_id composite key).
    Each timestep contains [q, dq].
    """
    X_list, y_list = [], []
    skipped = 0

    for vid in sorted(df['video_id'].unique()):
        vdf = df[df['video_id'] == vid].reset_index(drop=True)
        if len(vdf) < seq_len:
            skipped += 1
            continue
        q  = vdf[STATE_COLS].values.astype(np.float32)
        dq = vdf[VEL_COLS].values.astype(np.float32)
        qv = np.concatenate([q, dq], axis=1)
        a  = vdf[ACTION_COLS].values.astype(np.float32)
        for i in range(seq_len - 1, len(vdf)):
            X_list.append(qv[i - seq_len + 1 : i + 1])
            y_list.append(a[i])

    if skipped:
        print(f'  [WARN] {skipped} video(s) skipped (fewer than {seq_len} frames)')

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------
def _train_fold(df_trn, df_val, device, fold_idx, output_dir):
    X_trn, y_trn = build_sequences(df_trn, SEQ_LEN)
    X_val, y_val = build_sequences(df_val, SEQ_LEN)

    if len(X_trn) == 0 or len(X_val) == 0:
        print(f'  [SKIP] Not enough sequences for fold {fold_idx}.')
        return float('inf'), [], []

    scaler     = StandardScaler()
    X_trn_flat = X_trn.reshape(-1, N_INPUT)
    scaler.fit(X_trn_flat)
    X_trn_n = scaler.transform(X_trn_flat).reshape(-1, SEQ_LEN, N_INPUT).astype(np.float32)
    X_val_n = scaler.transform(X_val.reshape(-1, N_INPUT)).reshape(-1, SEQ_LEN, N_INPUT).astype(np.float32)

    scaler_path = output_dir / f'scaler_fold_{fold_idx}.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    X_trn_t = torch.tensor(X_trn_n).to(device)
    y_trn_t = torch.tensor(y_trn).to(device)
    X_val_t = torch.tensor(X_val_n).to(device)
    y_val_t = torch.tensor(y_val).to(device)

    trn_loader = DataLoader(TensorDataset(X_trn_t, y_trn_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

    model      = BCPolicyGRU(N_INPUT, HIDDEN_SIZE, N_JOINTS, N_LAYERS).to(device)
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
                'input_dim':   N_INPUT,
                'output_dim':  N_JOINTS,
                'hidden_size': HIDDEN_SIZE,
                'n_layers':    N_LAYERS,
                'seq_len':     SEQ_LEN,
                'val_loss':    val_loss,
                'hparams': {
                    'hidden_size': HIDDEN_SIZE,
                    'n_layers':    N_LAYERS,
                    'seq_len':     SEQ_LEN,
                    'lr':          LR,
                    'batch_size':  BATCH_SIZE,
                    'noise_std':   NOISE_STD,
                },
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
    ax.set_title('GRU — K-fold training curves')
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
    parser = argparse.ArgumentParser(description='Train BC GRU (k-fold) for one exercise.')
    parser.add_argument('exercise', type=int, help='Exercise number (e.g. 1)')
    parser.add_argument('--n-demos', type=int, default=55, choices=[10,25,55])
    args = parser.parse_args()

    split_dir    = DATA_ROOT / 'dataset' / f'exercise_{args.exercise:03d}' / split_name(args.n_demos)
    dataset_path = split_dir / 'bc_dataset.csv'
    output_dir    = split_dir / 'GRU'   # o GRU / Transformer
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading {dataset_path} ...')
    if not dataset_path.exists():
        print(f'Error: dataset not found → {dataset_path}')
        return

    df = pd.read_csv(dataset_path)
    print(f'  {len(df)} samples across {df["video"].nunique()} video(s)\n')

    # Composite key: uniquely identifies each video across all subjects
    df['video_id'] = df['subject'].astype(str) + '_' + df['video'].astype(str)

    np.random.seed(RANDOM_SEED)
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
    print(f'Model: GRU(input={N_INPUT}, hidden={HIDDEN_SIZE}, layers={N_LAYERS}) → Linear → {N_JOINTS}')
    print(f'Parameters: {sum(p.numel() for p in BCPolicyGRU(N_INPUT, HIDDEN_SIZE, N_JOINTS, N_LAYERS).parameters()):,}\n')

    fold_val_losses = []
    all_train_curves, all_val_curves = [], []

    for fold_idx in range(K_FOLDS):
        val_ids = set(folds[fold_idx].tolist())
        trn_ids = set(video_ids.tolist()) - val_ids

        df_trn = df[df['video_id'].isin(trn_ids)]
        df_val = df[df['video_id'].isin(val_ids)]

        print(f'{"="*50}')
        print(f'Fold {fold_idx}  |  train: {sorted(trn_ids)}  val: {sorted(val_ids)}')
        print(f'{"="*50}')
        print(f"{'Epoch':>6}  {'Train loss':>12}  {'Val loss':>12}")
        print('-' * 36)

        best_val, tl, vl = _train_fold(df_trn, df_val, device, fold_idx, output_dir)

        fold_val_losses.append(best_val)
        all_train_curves.append(tl)
        all_val_curves.append(vl)
        print()

    val_arr = np.array(fold_val_losses)
    print(f'\n{"="*50}')
    print(f'K-fold summary  [GRU]')
    print(f'{"="*50}')
    for i, v in enumerate(fold_val_losses):
        print(f'  Fold {i} val loss : {v:.6f}')
    print(f'  Mean ± std      : {val_arr.mean():.6f} ± {val_arr.std():.6f}')
    print(f'  Best fold       : {int(np.argmin(val_arr))}')
    print(f'{"="*50}\n')

    _save_loss_curve(all_train_curves, all_val_curves, output_dir / 'loss_curve.png')


if __name__ == '__main__':
    main()