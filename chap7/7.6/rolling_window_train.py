import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
from paths import stock_file

# ----------------------------
# Fix the random seed
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ========================
# TimeSeriesScaler
# ========================
class TimeSeriesScaler:
    def __init__(self):
        self.feature_mean = None
        self.feature_std = None
        self.target_mean = None
        self.target_std = None
        self.fitted = False

    def fit(self, features, targets):
        """Compute mean and standard deviation from the training set."""
        self.feature_mean = features.mean(axis=0, keepdims=True)
        self.feature_std = features.std(axis=0, keepdims=True) + 1e-8
        self.target_mean = targets.mean()
        self.target_std = targets.std() + 1e-8
        self.fitted = True
        return self

    def transform(self, features, targets):
        """Standardize features and targets using the training set's statistics."""
        assert self.fitted, "Scaler has not been fitted yet."
        features_scaled = (features - self.feature_mean) / self.feature_std
        targets_scaled = (targets - self.target_mean) / self.target_std
        return features_scaled, targets_scaled

    def fit_transform(self, features, targets):
        """Fit then transform."""
        self.fit(features, targets)
        return self.transform(features, targets)

    def inverse_transform_targets(self, targets_scaled):
        """Invert standardization on predictions or labels."""
        return targets_scaled * self.target_std + self.target_mean

# ========================
# create_dataloader
# ========================
def create_dataloader(data, seq_len, batch_size, scaler=None, is_train=False):
    df = data.select_dtypes(include=[np.number]).copy()
    vals = df.to_numpy(dtype=np.float32)

    features = vals[:, :-1]
    targets = vals[:, -1]

    # scaler must be provided (we follow your previous pipeline which supplies it)
    if scaler is None:
        scaler = TimeSeriesScaler()

    # ===== Training set: fit + transform =====
    if is_train:
        features, targets = scaler.fit_transform(features, targets)
    else:
        features, targets = scaler.transform(features, targets)

    vals_scaled = np.hstack([features, targets.reshape(-1, 1)])

    # Build sequences
    X, y = [], []
    for i in range(len(vals_scaled) - seq_len):
        X.append(vals_scaled[i:i + seq_len, :-1])
        y.append(vals_scaled[i + seq_len, -1])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)

    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

# ========================
# Transformer model
# ========================
class Transformer(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, dropout, output_size):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_size)

        # Weight initialization
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        out = self.fc_out(x[:, -1, :])
        return out

# ========================
# Evaluation metric
# ========================
def direction_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred))

# ========================
# Training function
# ========================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience, device):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # ---------- Train ----------
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader) if len(train_loader) > 0 else 1.0
        train_losses.append(train_loss)

        # ---------- Validate ----------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader) if len(val_loader) > 0 else 1.0
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Adjust the learning rate
        if scheduler is not None:
            scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "best_model.pth")  # save the best model
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses

# ========================
# Rolling (expanding) window training
# ========================
def rolling_window_train(df, seq_len, batch_size, model_cls, model_kwargs,
                         lr, epochs, patience, device,
                         train_ratio=0.8, val_ratio=0.1):
    """
    Expanding-window (rolling) training.
    - df: pandas.DataFrame (whole series)
    - model_cls: model class (e.g. Transformer)
    - model_kwargs: dict passed to model_cls(**model_kwargs)
    Returns: best_model (with best window weights), all_train_losses, all_val_losses, best_scaler
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_size = int(n * val_ratio)
    if val_size < 1:
        raise ValueError("val_ratio too small or df too short.")

    best_val_loss = float("inf")
    best_wts = None
    best_scaler = None
    all_train_losses = []
    all_val_losses = []

    split = train_end
    # Each step expands the training set up to split; the validation set is [split: split+val_size)
    while split <= n - val_size:
        train_df = df.iloc[:split].copy()
        val_df = df.iloc[split: split + val_size].copy()

        # Each window gets its own scaler, fit on that window's training portion
        scaler = TimeSeriesScaler()
        train_loader = create_dataloader(train_df, seq_len, batch_size, scaler, is_train=True)
        val_loader = create_dataloader(val_df, seq_len, batch_size, scaler, is_train=False)

        # new model for this window
        model = model_cls(**model_kwargs).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

        model, train_losses, val_losses = train_model(model, train_loader, val_loader,
                                                      criterion, optimizer, scheduler,
                                                      epochs, patience, device)

        all_train_losses.extend(train_losses)
        all_val_losses.extend(val_losses)

        # track global best by min val_loss in this window
        if len(val_losses) > 0:
            min_val = min(val_losses)
            if min_val < best_val_loss:
                best_val_loss = min_val
                best_wts = copy.deepcopy(model.state_dict())
                best_scaler = copy.deepcopy(scaler)

        # move window forward by val_size
        split += val_size

    # assemble final model and load best weights
    final_model = model_cls(**model_kwargs).to(device)
    if best_wts is not None:
        final_model.load_state_dict(best_wts)
        torch.save(best_wts, "best_model.pth")
    else:
        print("Warning: no best model found during rolling windows.")

    return final_model, all_train_losses, all_val_losses, best_scaler

# ========================
# Fixed-length sliding window training
# ========================
def fixed_window_train(df, seq_len, batch_size, model_cls, model_kwargs,
                       lr, epochs, patience, device,
                       train_ratio=0.6, val_ratio=0.2, step_size=1):
    """
    Fixed-length sliding window training.
    - window_size = int((train_ratio + val_ratio) * len(df))
    - Slide the window with step_size (default 1)
    Returns: best_model, all_train_losses, all_val_losses, best_scaler
    """
    n = len(df)
    window_size = int((train_ratio + val_ratio) * n)
    if window_size <= seq_len:
        raise ValueError("Window too small relative to seq_len.")

    # inside each window, compute partition of train/val (proportionally)
    total_prop = train_ratio + val_ratio
    train_portion = train_ratio / total_prop
    # train_size and val_size within the window
    train_size_win = int(window_size * train_portion)
    val_size_win = window_size - train_size_win
    if val_size_win < 1 or train_size_win < 1:
        raise ValueError("train_ratio/val_ratio produce too small splits in the window.")

    best_val_loss = float("inf")
    best_model = None
    best_scaler = None
    train_losses_all = []
    val_losses_all = []

    # ensure last index leaves room for seq_len when creating sequences
    max_start = n - window_size - seq_len + 1
    if max_start < 0:
        max_start = 0

    for start in range(0, max_start + 1, step_size):
        end = start + window_size
        df_window = df.iloc[start:end].copy()

        # window internal split
        train_df = df_window.iloc[:train_size_win]
        val_df = df_window.iloc[train_size_win: train_size_win + val_size_win]

        # use TimeSeriesScaler + create_dataloader (consistent with pipeline)
        scaler = TimeSeriesScaler()
        train_loader = create_dataloader(train_df, seq_len, batch_size, scaler, is_train=True)
        val_loader = create_dataloader(val_df, seq_len, batch_size, scaler, is_train=False)

        model = model_cls(**model_kwargs).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

        model, train_losses, val_losses = train_model(model, train_loader, val_loader,
                                                      criterion, optimizer, scheduler,
                                                      epochs, patience, device)

        train_losses_all.extend(train_losses)
        val_losses_all.extend(val_losses)

        if len(val_losses) > 0 and min(val_losses) < best_val_loss:
            best_val_loss = min(val_losses)
            best_model = copy.deepcopy(model)
            best_scaler = copy.deepcopy(scaler)

        print(f"Window [{start}:{end}] finished. Current best val_loss={best_val_loss:.6f}")

    if best_model is None:
        print("Warning: no best model found during fixed-window training.")
        final_model = model_cls(**model_kwargs).to(device)
    else:
        final_model = best_model
        torch.save(final_model.state_dict(), "best_model.pth")

    return final_model, train_losses_all, val_losses_all, best_scaler

# ========================
# Unified entry
# ========================
def train_with_mode(df, mode, seq_len, batch_size, model_cls, model_kwargs,
                    lr, epochs, patience, device,
                    train_ratio=0.8, val_ratio=0.1, step_size=1):
    """
    Unified entry:
      mode: 'rolling' (expanding) or 'fixed' (sliding fixed-length)
    Returns: model, train_losses, val_losses, best_scaler
    """
    mode = mode.lower()
    if mode == "rolling":
        return rolling_window_train(df, seq_len, batch_size, model_cls, model_kwargs,
                                    lr, epochs, patience, device, train_ratio, val_ratio)
    elif mode == "fixed":
        return fixed_window_train(df, seq_len, batch_size, model_cls, model_kwargs,
                                  lr, epochs, patience, device, train_ratio, val_ratio, step_size)
    else:
        raise ValueError("mode must be 'rolling' or 'fixed'")

# ========================
# Main
# ========================
if __name__ == "__main__":
    df = pd.read_csv(stock_file('688256.SH.csv'), index_col='trade_date')

    # ===== 1. Parameters =====
    seq_len = 90  # lookback window length
    batch_size = 32  # batch size
    input_size = df.shape[1] - 1  # number of features
    output_size = 1  # forecast horizon
    d_model = 128  # model dimension
    num_heads = 4  # number of attention heads
    num_layers = 2  # number of encoder layers
    dropout = 0.1  # dropout rate
    lr = 1e-4  # learning rate
    epochs = 100  # number of training epochs
    patience = 20  # early-stopping patience
    train_ratio = 0.8  # training-set ratio (for window splitting)
    val_ratio = 0.1  # validation-set ratio (for window splitting)

    # ===== 2. Device =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ===== 3. Pack model constructor args into a dict (passed to model_cls) =====
    model_params = dict(
        input_size=input_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        output_size=output_size
    )

    # ===== 4. Choose the training mode: 'fixed' or 'rolling' =====
    mode = "fixed"  # change to "rolling" to switch modes

    # ===== 5. Call the unified entry point to train (rolling or fixed) =====
    model, train_losses, val_losses, best_scaler = train_with_mode(
        df, mode, seq_len, batch_size,
        Transformer, model_params,
        lr, epochs, patience, device,
        train_ratio=train_ratio, val_ratio=val_ratio, step_size=1
    )

    # ===== 6. Build the test-set loader (using best_scaler to keep scale consistent) =====
    # test_df with same splitting logic as before: reserve last portion for test
    train_size = int(len(df) * train_ratio)
    val_size = int(len(df) * val_ratio)
    test_df = df.iloc[train_size + val_size:]

    if best_scaler is None:
        # fallback: fit scaler on original train split to keep behavior consistent
        best_scaler = TimeSeriesScaler()
        train_df_for_scaler = df.iloc[:train_size]
        features = train_df_for_scaler.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)[:, :-1]
        targets = train_df_for_scaler.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)[:, -1]
        best_scaler.fit(features, targets)

    test_loader = create_dataloader(test_df, seq_len, batch_size, best_scaler, is_train=False)

    # ===== 7. Test evaluation =====
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            preds_real = best_scaler.inverse_transform_targets(preds)
            targets_real = best_scaler.inverse_transform_targets(y_batch.cpu().numpy())

            all_preds.extend(preds_real.flatten())
            all_targets.extend(targets_real.flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # ===== 8. Compute metrics =====
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    da = direction_accuracy(all_targets, all_preds)

    print(f"\nTest MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Test Direction Accuracy: {da:.4f}")

    # ===== 9. Plot =====
    plt.figure(1, figsize=(12, 6))
    plt.plot(all_targets, label="True", alpha=0.7)
    plt.plot(all_preds, label="Predicted", alpha=0.7)
    plt.legend()
    plt.title("True vs Predicted on Test Set")
    plt.xlabel("Samples")
    plt.ylabel("Target Value")
    plt.grid(True)

    plt.figure(2, figsize=(12, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Train and Validation Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
