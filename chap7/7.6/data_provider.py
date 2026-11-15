import numpy as np
import torch


#========================
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



 #=========================
def create_dataloader(data, seq_len, batch_size, scaler=None, is_train=False):
    df = data.select_dtypes(include=[np.number]).copy()
    vals = df.to_numpy(dtype=np.float32)

    features = vals[:, :-1]
    targets = vals[:, -1]

    # ===== Training set: fit + transform =====
    if is_train:
        features, targets = scaler.fit_transform(features, targets)
    else:
        features, targets = scaler.transform(features, targets)

    vals = np.hstack([features, targets.reshape(-1, 1)])

    # Build sequences
    X, y = [], []
    for i in range(len(vals) - seq_len):
        X.append(vals[i:i + seq_len, :-1])
        y.append(vals[i + seq_len, -1])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)

    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
