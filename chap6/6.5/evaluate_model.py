import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate_model(model, test_loader, scaler, device='cpu'):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred_batch = model(X_batch)
            y_true.append(y_batch.cpu().numpy())
            y_pred.append(y_pred_batch.cpu().numpy())

    y_true = np.vstack(y_true)  # stack along axis 0 (rows); the last batch may have fewer samples
    y_pred = np.vstack(y_pred)

    # Invert standardization
    y_true_original = scaler.inverse_transform(y_true)
    y_pred_original = scaler.inverse_transform(y_pred)

    # Compute MSE and MAE
    metrics ={}
    mse = mean_squared_error(y_true_original.reshape(-1), y_pred_original.reshape(-1))
    mae = mean_absolute_error(y_true_original.reshape(-1), y_pred_original.reshape(-1))
    rmse = np.sqrt(mse)
    mape = np.mean(
        np.abs((y_true_original.reshape(-1) - y_pred_original.reshape(-1)) / y_true_original.reshape(-1))) * 100

    metrics['close'] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    return metrics, y_true_original, y_pred_original
