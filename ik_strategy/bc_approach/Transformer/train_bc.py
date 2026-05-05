'''
Transformer/train_bc.py
=============================================================================
Trains the Behavioral Cloning Transformer for a single exercise using k-fold
cross validation. Produces K independent models, one per fold.

Architecture:
  - Positional encoding on input sequence
  - TransformerEncoder (N_HEADS attention heads, N_LAYERS layers)
  - Linear head on last timestep → Δq

Input  : sequence of [q, dq] values, shape (SEQ_LEN, 16)
Output : Δq(t) — joint delta (8 values)

The Transformer processes all timesteps in parallel via self-attention,
weighting each past state dynamically based on learned relevance.
A causal mask ensures the model only attends to past timesteps.

K-fold: same video-level split as MLP and GRU.
Ensemble: all K models are averaged at inference (see test_bc.py).

--- Input ---
  data/dataset/exercise_XXX/bc_dataset.csv

--- Output ---
  data/dataset/exercise_XXX/Transformer/bc_model_fold_0.pth  ...
  data/dataset/exercise_XXX/Transformer/scaler_fold_0.pkl    ...
  data/dataset/exercise_XXX/Transformer/loss_curve.png

Usage:
  py -m bc_approach.Transformer.train_bc 1
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

from utilities.config import DATA_ROOT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
JOINT_COLS = [
    'r_shoulder_pitch', 'r_shoulder_roll', 'r_arm_yaw', 'r_elbow_pitch',
    'l_shoulder_pitch', 'l_shoulder_roll', 'l_arm_yaw', 'l_elbow_pitch',
]
N_JOINTS    = len(JOINT_COLS)
STATE_COLS  = [f'q_{j}'  for j in JOINT_COLS]
VEL_COLS    = [f'dq_{j}' for j in JOINT_COLS]
ACTION_COLS = [f'act_{j}' for j in JOINT_COLS]

# Each timestep contains [q, dq] → 16 features
N_INPUT       = N_JOINTS * 2  # 16

SEQ_LEN       = 20
K_FOLDS_RATIO = 0.25

# Transformer hyperparameters
# D_MODEL must be divisible by N_HEADS
D_MODEL   = 64    # embedding dimension
N_HEADS   = 4     # attention heads
N_LAYERS  = 2     # transformer encoder layers
D_FF      = 128   # feedforward hidden dim inside each transformer block
DROPOUT   = 0.1

BATCH_SIZE  = 64
LR          = 1e-3
PATIENCE    = 20
MAX_EPOCHS  = 500
RANDOM_SEED = 42
NOISE_STD   = 0.1


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding — adds position information to the
    input embeddings so the Transformer knows the order of timesteps.
    """
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BCPolicyTransformer(nn.Module):
    """
    Causal Transformer policy: sequence of [q, dq] (SEQ_LEN, 16) -> Δq (8).

    Input is projected to D_MODEL dimensions, positional encoding is added,
    then processed by a TransformerEncoder with a causal mask.
    The output at the last timestep is passed to a linear head.
    """
    def __init__(self, input_dim: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head        = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        seq_len = x.size(1)

        # Causal mask: position i can only attend to positions <= i
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x.device
        )
        x = self.input_proj(x)          # (batch, seq_len, d_model)
        x = self.pos_enc(x)             # add positional encoding
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        return self.head(x[:, -1, :])   # last timestep → (batch, output_dim)


# ---------------------------------------------------------------------------
# Dataset builder (same as GRU)
# ---------------------------------------------------------------------------
def build_sequences(df: pd.DataFrame, seq_len: int):
    """
    Creates sliding-window sequences from bc_dataset.csv.
    Windows never cross video boundaries.
    Each timestep contains [q, dq] — 16 features.
    """
    X_list, y_list = [], []
    skipped = 0

    for video_id in sorted(df['video'].unique()):
        vdf = df[df['video'] == video_id].reset_index(drop=True)
        if len(vdf) < seq_len:
            skipped += 1
            continue
        q  = vdf[STATE_COLS].values.astype(np.float32)
        dq = vdf[VEL_COLS].values.astype(np.float32)
        qv = np.concatenate([q, dq], axis=1)             # (N, 16)
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
    """Trains one fold and saves model + scaler. Returns best val loss."""

    X_trn, y_trn = build_sequences(df_trn, SEQ_LEN)
    X_val, y_val = build_sequences(df_val, SEQ_LEN)

    if len(X_trn) == 0 or len(X_val) == 0:
        print(f'  [SKIP] Not enough sequences for fold {fold_idx}.')
        return float('inf'), [], []

    # Normalise
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

    model = BCPolicyTransformer(
        input_dim=N_INPUT, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, output_dim=N_JOINTS, dropout=DROPOUT
    ).to(device)

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
                'd_model':     D_MODEL,
                'n_heads':     N_HEADS,
                'n_layers':    N_LAYERS,
                'd_ff':        D_FF,
                'dropout':     DROPOUT,
                'seq_len':     SEQ_LEN,
                'val_loss':    val_loss,
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
    ax.set_title('Transformer — K-fold training curves')
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
    parser = argparse.ArgumentParser(description='Train BC Transformer (k-fold) for one exercise.')
    parser.add_argument('exercise', type=int, help='Exercise number (e.g. 1)')
    args = parser.parse_args()

    exercise_name = f'exercise_{args.exercise:03d}'
    dataset_path  = DATA_ROOT / 'dataset' / exercise_name / 'bc_dataset.csv'
    output_dir    = DATA_ROOT / 'dataset' / exercise_name / 'Transformer'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading {dataset_path} ...')
    if not dataset_path.exists():
        print(f'Error: dataset not found → {dataset_path}')
        return

    df = pd.read_csv(dataset_path)
    print(f'  {len(df)} samples across {df["video"].nunique()} video(s)\n')

    np.random.seed(RANDOM_SEED)
    video_ids = df['video'].unique()
    np.random.shuffle(video_ids)
    K_FOLDS = max(2, round(len(video_ids) * K_FOLDS_RATIO))
    folds   = np.array_split(video_ids, K_FOLDS)

    print(f'K-fold cross validation  (K={K_FOLDS}, ratio={K_FOLDS_RATIO})')
    print(f'  Total videos : {len(video_ids)}')
    for i, fold in enumerate(folds):
        print(f'  Fold {i}       : val videos {sorted(fold.tolist())}')
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Model: Transformer(d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, d_ff={D_FF})')
    _dummy = BCPolicyTransformer(N_INPUT, D_MODEL, N_HEADS, N_LAYERS, D_FF, N_JOINTS, DROPOUT)
    print(f'Parameters: {sum(p.numel() for p in _dummy.parameters()):,}\n')

    fold_val_losses = []
    all_train_curves, all_val_curves = [], []

    for fold_idx in range(K_FOLDS):
        val_videos = set(folds[fold_idx].tolist())
        trn_videos = set(video_ids.tolist()) - val_videos

        df_trn = df[df['video'].isin(trn_videos)]
        df_val = df[df['video'].isin(val_videos)]

        print(f'{"="*50}')
        print(f'Fold {fold_idx}  |  train: {sorted(trn_videos)}  val: {sorted(val_videos)}')
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
    print(f'K-fold summary  [Transformer]')
    print(f'{"="*50}')
    for i, v in enumerate(fold_val_losses):
        print(f'  Fold {i} val loss : {v:.6f}')
    print(f'  Mean ± std      : {val_arr.mean():.6f} ± {val_arr.std():.6f}')
    print(f'  Best fold       : {int(np.argmin(val_arr))}')
    print(f'{"="*50}\n')

    _save_loss_curve(all_train_curves, all_val_curves, output_dir / 'loss_curve.png')


if __name__ == '__main__':
    main()