'''
train_bc.py
=============================================================================
Trains a Behavioral Cloning (BC) model for a single exercise.

The model is a Multi-Layer Perceptron (MLP) that learns to map the current
robot state to the action (joint angle delta) to apply next:

Architecture:
  input  (state_dim)
      -> Linear + ReLU   (256 neurons)
      -> Linear + ReLU   (256 neurons)
      -> Linear          (action_dim)
  loss      : MSE
  optimizer : Adam (lr=0.001)

Training procedure:
  1. Load bc_dataset.csv for the given exercise.
  2. Split 80/20 into training and validation sets (random, reproducible).
  3. Fit a StandardScaler on training state features only -> scaler.pkl
  4. Train with mini-batches (size 64), shuffling each epoch.
  5. Early stopping: save the model whenever validation loss improves; stop if it has not improved for `patience` epochs.
  6. Save best model weights -> bc_model.pth

Data augmentation:
  Gaussian noise (std=0.5) is added to state features during training.
  This simulates the distribution shift that occurs at inference time
  (autoregressive loop produces slightly imperfect states) and reduces
  compounding errors.

Output:
  data/dataset/exercise_XXX/bc_model.pth    <- model weights (PyTorch)
  data/dataset/exercise_XXX/scaler.pkl      <- fitted StandardScaler
  data/dataset/exercise_XXX/training_log.csv <- loss per epoch

Usage:
  # train on exercise 1 with default parameters
  python train_bc.py --exercise 1

  # custom hyperparameters
  python train_bc.py --exercise 1 --epochs 300 --patience 30 --lr 0.0005

  # disable data augmentation noise
  python train_bc.py --exercise 1 --noise 0.0

  # run inference loop on a saved model (quick sanity check)
  python train_bc.py --exercise 1 --infer-only
'''

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from config import DATA_ROOT

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS      = 300
DEFAULT_PATIENCE    = 20
DEFAULT_LR          = 0.001
DEFAULT_BATCH_SIZE  = 64
DEFAULT_NOISE_STD   = 0.1    # Gaussian noise added to state during training
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_N_LAYERS    = 2
RANDOM_SEED         = 42


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BCModel(nn.Module):
    '''
    Multi-Layer Perceptron for Behavioral Cloning.

    Architecture: input_dim -> [hidden_size x ReLU] x n_layers -> output_dim

    Parameters:
        input_dim   : state dimension (automatically inferred from dataset)
        output_dim  : action dimension (automatically inferred from dataset)
        hidden_size : number of neurons per hidden layer (default: 256)
        n_layers    : number of hidden layers (default: 2)
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = DEFAULT_HIDDEN_SIZE, n_layers: int = DEFAULT_N_LAYERS):
        super().__init__()

        layers = []
        in_features = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_features, hidden_size), nn.ReLU()]
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def _identify_columns(df: pd.DataFrame):
    '''
    Automatically identifies state and action columns from the DataFrame.

    State columns  : 'state_*', 'vel_*', and optionally 'head_x/y/z'
    Action columns : 'action_*'

    Returns:
        state_cols : list of state column names
        action_cols: list of action column names
    '''
    state_cols  = (
        [c for c in df.columns if c.startswith('state_')]
        + [c for c in df.columns if c.startswith('vel_')]
        + [c for c in df.columns if c in ('head_x', 'head_y', 'head_z')]
    )
    action_cols = [c for c in df.columns if c.startswith('action_')]

    return state_cols, action_cols


def _load_dataset(dataset_path: Path):
    '''
    Loads bc_dataset.csv and returns:
      - X : (N, state_dim) numpy array -> state features
      - y : (N, action_dim) numpy array -> action targets
      - state_cols, action_cols : column name lists (for logging)
    '''
    df = pd.read_csv(dataset_path)
    print(f'  Loaded {len(df)} samples from {dataset_path}')

    state_cols, action_cols = _identify_columns(df)

    if not state_cols:
        raise ValueError('No state columns found. Check bc_dataset.csv.')
    if not action_cols:
        raise ValueError('No action columns found. Check bc_dataset.csv.')

    # Drop rows with any NaN in state or action
    all_cols = state_cols + action_cols
    n_before = len(df)
    df = df.dropna(subset=all_cols).reset_index(drop=True)
    if len(df) < n_before:
        print(f'  Dropped {n_before - len(df)} rows with NaN values.')

    X = df[state_cols].values.astype(np.float32)
    y = df[action_cols].values.astype(np.float32)

    print(f'  State  dimension : {X.shape[1]}  columns: {state_cols[:3]}...')
    print(f'  Action dimension : {y.shape[1]}  columns: {action_cols[:3]}...')

    return X, y, state_cols, action_cols


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _train(exercise_num: int, dataset_root: Path, epochs: int, patience: int, lr: float, batch_size: int, noise_std: float, hidden_size: int, n_layers: int) -> None:
    exercise_name = f'exercise_{exercise_num:03d}'
    output_dir    = dataset_root / exercise_name
    dataset_path  = output_dir / 'bc_dataset.csv'

    if not dataset_path.exists():
        print(f'Error: bc_dataset.csv not found -> {dataset_path}')
        print('Run build_dataset.py first.')
        return

    print(f'\n{"="*60}')
    print(f'  Training BC model -> Exercise {exercise_num:03d}')
    print(f'{"="*60}')

    # Load data
    X, y, state_cols, action_cols = _load_dataset(dataset_path)

    # Train / validation split
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)
    print(f'\n  Train samples      : {len(X_tr)}')
    print(f'  Validation samples : {len(X_val)}')

    # Normalise state features
    # Fit ONLY on training set to avoid data leakage
    scaler = StandardScaler()
    X_tr_scaled  = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    scaler_path = output_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f'  Scaler saved -> {scaler_path}')

    # Build PyTorch datasets
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device : {device}')

    X_tr_t   = torch.tensor(X_tr_scaled)
    y_tr_t   = torch.tensor(y_tr.astype(np.float32))
    X_val_t  = torch.tensor(X_val_scaled)
    y_val_t  = torch.tensor(y_val.astype(np.float32))

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size = batch_size, shuffle = True)

    # Model, loss, optimizer
    state_dim  = X.shape[1]
    action_dim = y.shape[1]

    model = BCModel(state_dim, action_dim, hidden_size, n_layers).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\n  Architecture : {state_dim} -> '
          f'{" -> ".join([str(hidden_size)] * n_layers)} -> {action_dim}')
    print(f'  Parameters   : {total_params:,}')

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print(f'\n  Epochs={epochs}  Patience={patience}  '
          f'LR={lr}  BatchSize={batch_size}  NoiseStd={noise_std}')
    print(f'  {"Epoch":>6}  {"Train Loss":>12}  {"Val Loss":>12}  {"Best":>6}')
    print(f'  {"-"*44}')

    best_val_loss   = float('inf')
    patience_counter = 0
    log_rows        = []
    model_path      = output_dir / 'bc_model.pth'

    X_val_dev = X_val_t.to(device)
    y_val_dev = y_val_t.to(device)

    for epoch in range(1, epochs + 1):

        # Training phase
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Data augmentation: Gaussian noise on state
            if noise_std > 0:
                X_batch = X_batch + torch.randn_like(X_batch) * noise_std

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X_batch)

        train_loss /= len(X_tr)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss  = criterion(model(X_val_dev), y_val_dev).item()

        # Early stopping
        is_best = val_loss < best_val_loss
        if is_best:
            best_epoch = epoch
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                'epoch'       : epoch,
                'model_state' : model.state_dict(),
                'state_dim'   : state_dim,
                'action_dim'  : action_dim,
                'hidden_size' : hidden_size,
                'n_layers'    : n_layers,
                'state_cols'  : state_cols,
                'action_cols' : action_cols,
                'val_loss'    : best_val_loss,
            }, model_path)
        else:
            patience_counter += 1

        log_rows.append({
            'epoch'      : epoch,
            'train_loss' : train_loss,
            'val_loss'   : val_loss,
            'best'       : is_best,
        })

        # Print every 10 epochs and at the last epoch
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            marker = ' <--' if is_best else ''
            print(f'  {epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}'
                  f'  {"YES" if is_best else "":>6}{marker}')

        if patience_counter >= patience:
            print(f'\n  Early stopping at epoch {epoch} '
                  f'(no improvement for {patience} epochs).')
            break

    # Save training log
    log_path = output_dir / 'training_log.csv'
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    print(f'\n  Best val loss : {best_val_loss:.6f}  (epoch {best_epoch})')
    print(f'  Model saved   -> {model_path}')
    print(f'  Log saved     -> {log_path}')


# ---------------------------------------------------------------------------
# Inference sanity check
# ---------------------------------------------------------------------------
def _infer(exercise_num: int, dataset_root: Path, n_steps: int = 50) -> None:
    '''
    Loads a saved model and runs the autoregressive loop for n_steps
    starting from the rest pose. Prints the first few predicted deltas
    as a quick sanity check; does not connect to Reachy.
    '''
    exercise_name = f'exercise_{exercise_num:03d}'
    output_dir    = dataset_root / exercise_name
    model_path    = output_dir / 'bc_model.pth'
    scaler_path   = output_dir / 'scaler.pkl'

    if not model_path.exists():
        print(f'Error: bc_model.pth not found -> {model_path}')
        return
    if not scaler_path.exists():
        print(f'Error: scaler.pkl not found -> {scaler_path}')
        return

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = BCModel(checkpoint['state_dim'], checkpoint['action_dim'], checkpoint['hidden_size'], checkpoint['n_layers'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    state_dim  = checkpoint['state_dim']
    action_dim = checkpoint['action_dim']

    print(f'\n  Inference check - {n_steps} autoregressive steps')
    print(f'  State dim: {state_dim}  Action dim: {action_dim}')

    # Initial state: rest pose + zero velocity + neutral head
    state = np.zeros(state_dim, dtype=np.float32)
    # Reachy rest pose (degrees)
    rest_pose = np.array([
         0., -5.,  0., -90., 0., 0., 0., 0.,   # right arm
         0.,  5.,  0., -90., 0., 0., 0., 0.,   # left arm
    ], dtype=np.float32)
    state[:16] = rest_pose          # q(0)
    state[16:32] = 0.0              # vel(0) = zero
    if state_dim > 32:
        state[32:35] = [1.0, 0.0, 0.15]   # neutral head gaze

    trajectory = [state[:16].copy()]

    print(f'\n  {"Step":>5}  {"Max |delta|":>12}  {"Shoulder pitch":>16}')
    print(f'  {"-"*38}')

    for step in range(n_steps):
        state_norm = scaler.transform([state])
        x_t = torch.tensor(state_norm, dtype=torch.float32)

        with torch.no_grad():
            delta = model(x_t).numpy()[0]   # shape: (action_dim,)

        # Apply delta to joint angles only (first 16 values)
        q_new   = state[:16] + delta[:16]
        vel_new = q_new - state[:16]

        # Update state
        state[:16]  = q_new
        state[16:32] = vel_new
        # head gaze not updated in this sanity check

        trajectory.append(q_new.copy())

        if step < 5 or step == n_steps - 1:
            print(f'  {step+1:>5}  {np.max(np.abs(delta)):>12.4f}  '
                  f'{q_new[0]:>16.4f}°')

    trajectory = np.array(trajectory)
    print(f'\n  Trajectory shape : {trajectory.shape}')
    print(f'  Joint range (r_shoulder_pitch): '
          f'{trajectory[:, 0].min():.1f}° to {trajectory[:, 0].max():.1f}°')
    print('  Inference check passed.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train a Behavioral Cloning MLP for one exercise.')
    parser.add_argument('--exercise', type=int, required=True,
        help='Exercise number to train on (required).')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
        help=f'Maximum training epochs (default: {DEFAULT_EPOCHS}).')
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE,
        help=f'Early stopping patience (default: {DEFAULT_PATIENCE}).')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
        help=f'Adam learning rate (default: {DEFAULT_LR}).')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
        help=f'Mini-batch size (default: {DEFAULT_BATCH_SIZE}).')
    parser.add_argument('--noise', type=float, default=DEFAULT_NOISE_STD,
        help=f'Std of Gaussian noise added to state during training '
             f'(default: {DEFAULT_NOISE_STD}). Set 0 to disable.')
    parser.add_argument('--hidden-size', type=int, default=DEFAULT_HIDDEN_SIZE,
        help=f'Neurons per hidden layer (default: {DEFAULT_HIDDEN_SIZE}).')
    parser.add_argument('--n-layers', type=int, default=DEFAULT_N_LAYERS,
        help=f'Number of hidden layers (default: {DEFAULT_N_LAYERS}).')
    parser.add_argument('--infer-only', action='store_true',
        help='Skip training and run inference sanity check on saved model.')
    args = parser.parse_args()

    dataset_root = DATA_ROOT / 'dataset'

    if args.infer_only:
        _infer(args.exercise, dataset_root)
    else:
        _train(
            exercise_num = args.exercise,
            dataset_root = dataset_root,
            epochs       = args.epochs,
            patience     = args.patience,
            lr           = args.lr,
            batch_size   = args.batch_size,
            noise_std    = args.noise,
            hidden_size  = args.hidden_size,
            n_layers     = args.n_layers,
        )


if __name__ == '__main__':
    main()