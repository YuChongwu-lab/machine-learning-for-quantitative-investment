import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from paths import stock_file



#Load data
def get_data(seq_len, pred_horizon):
    df = pd.read_excel(stock_file('000001SZ.xlsx'))
    data=df[['open','high','low','vol','close']]

    X, y = [], []
    for i in range(len(data)-seq_len-pred_horizon):
        X.append(data[i:i + seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_horizon]['close'])

    return np.array(X), np.array(y)



def prepare_data(X, y, train_ratio, val_ratio, batch_size):
    # Split the dataset
    n_samples = len(X)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # Standardize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Reshape to 2D for standardization
    X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
    X_val_reshaped = X_val.reshape(-1, X_val.shape[2])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[2])

    scaler_X.fit(X_train_reshaped)

    X_train_scaled = scaler_X.transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val_reshaped).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)

    scaler_y.fit(y_train)
    y_train_scaled = scaler_y.transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    # Build the data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val_scaled))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test_scaled))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, scaler_y

