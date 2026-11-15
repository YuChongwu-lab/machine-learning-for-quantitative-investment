"""Data loading and graph-dataset construction."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


def load_real_data(data_file_path: str,
                   max_assets: int = 30,
                   max_days: int = 500) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load the data, returning price (pivoted price matrix), bench (equal-weight
    benchmark level), and panel_data (the raw panel, aligned/filtered to match)."""
    print(f"Loading data: {data_file_path}")
    if not isinstance(data_file_path, str):
        raise TypeError("data_file_path should be a string path")
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")

    # Read the raw panel
    if data_file_path.endswith('.pkl'):
        panel_data = pd.read_pickle(data_file_path)
    elif data_file_path.endswith('.csv'):
        panel_data = pd.read_csv(data_file_path)
    else:
        panel_data = pd.read_excel(data_file_path)

    print(f"Data shape: {panel_data.shape}")
    required = ['trade_date', 'ts_code', 'close']
    miss = [c for c in required if c not in panel_data.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    if not pd.api.types.is_datetime64_any_dtype(panel_data['trade_date']):
        panel_data['trade_date'] = pd.to_datetime(panel_data['trade_date'])
    panel_data = panel_data.sort_values(['trade_date', 'ts_code'])

    # Pivot the prices
    price = panel_data.pivot(index='trade_date', columns='ts_code', values='close')
    price = price.sort_index().ffill().bfill()
    price = price.dropna(axis=0, how='all').dropna(axis=1, how='all')

    # Cap the scale (columns first, then rows)
    if max_assets is not None and max_assets > 0:
        price = price.iloc[:, :max_assets]
    if max_days is not None and max_days > 0:
        price = price.iloc[-max_days:]

    # Build the benchmark level from equal-weight returns
    eq_ret = price.pct_change().mean(axis=1).fillna(0.0)
    bench = [1000.0]
    for r in eq_ret[1:]:
        bench.append(bench[-1] * (1 + r))
    bench = pd.Series(bench, index=price.index, name='Benchmark')

    # Align panel_data's dates and stocks to match
    keep_dates = set(price.index)
    keep_codes = set(price.columns)
    mask = panel_data['trade_date'].isin(keep_dates) & panel_data['ts_code'].isin(keep_codes)
    panel_data = panel_data.loc[mask].copy()

    print(f"Processed price-data shape: {price.shape} (assets={price.shape[1]}, days={price.shape[0]})")
    return price, bench, panel_data


def create_enhanced_dataset(panel_data: pd.DataFrame,
                            seq_len: int = 10,
                            corr_window: int = 5,
                            corr_threshold: float = 0.3,
                            feature_cols: List[str] = None,
                            categorical_handling: str = 'drop') ->Tuple[List[Data],List[str]] :
    """
    Build a graph dataset using the feature columns already present in the raw data.
    - nodes = asset x time; node features = numeric feature columns from the raw
      panel, rolling-window standardized
    - temporal edges: a bidirectional edge between a given asset's t and t+1 (edge weight w=0)
    - correlation edges: a bidirectional edge between any asset pair whose rolling
      correlation over the past corr_window days exceeds the threshold,
      edge_attr = the signed correlation coefficient w in [-0.999, 0.999]
    """
    print(f"Creating the graph dataset... (corr_window={corr_window}, corr_threshold={corr_threshold})")

    required = ['trade_date', 'ts_code', 'close']
    miss = [c for c in required if c not in panel_data.columns]
    if miss:
        raise ValueError(f"panel_data is missing required columns: {miss}")

    if not pd.api.types.is_datetime64_any_dtype(panel_data['trade_date']):
        panel_data = panel_data.copy()
        panel_data['trade_date'] = pd.to_datetime(panel_data['trade_date'])
    panel_data = panel_data.sort_values(['trade_date', 'ts_code'])

    if feature_cols is None:
        feature_cols = [c for c in panel_data.columns if c not in ('trade_date', 'ts_code', 'close')]

    all_feat_cols = feature_cols[:]
    numeric_cols = panel_data[all_feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = sorted(list(set(all_feat_cols) - set(numeric_cols)))

    if non_numeric_cols:
        if categorical_handling not in ('drop', 'factorize'):
            raise ValueError(f"Unknown categorical_handling={categorical_handling} (must be 'drop' or 'factorize')")
        if categorical_handling == 'drop':
            print(f"[create] Found non-numeric columns (skipped): {non_numeric_cols[:8]}{' ...' if len(non_numeric_cols)>8 else ''}")
        else:
            print(f"[create] Found non-numeric columns (integer-encoded): {non_numeric_cols[:8]}{' ...' if len(non_numeric_cols)>8 else ''}")
            for col in non_numeric_cols:
                codes, _ = pd.factorize(panel_data[col], sort=True)
                codes = codes.astype(np.int32)
                codes[codes < 0] = 0
                panel_data[col + "_code"] = codes
            numeric_cols += [c + "_code" for c in non_numeric_cols]

    feature_cols = numeric_cols

    price = panel_data.pivot(index='trade_date', columns='ts_code', values='close')
    price = price.sort_index().replace([np.inf, -np.inf], np.nan).ffill().bfill()
    price = price.dropna(axis=0, how='all').dropna(axis=1, how='all')

    if price.shape[0] < seq_len + 2:
        print(f"[create] Not enough valid trading days: T={price.shape[0]}, seq_len+2={seq_len+2}")
        return [], []

    n_stocks = price.shape[1]
    if n_stocks == 0:
        print("[create] Zero assets, exiting.")
        return [], []

    dates = price.index
    returns = price.pct_change().fillna(0.0)

    feat_mats = []
    for col in feature_cols:
        mat = panel_data.pivot(index='trade_date', columns='ts_code', values=col)
        mat = mat.reindex(index=dates, columns=price.columns)
        mat = mat.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        feat_mats.append(mat.values)
    feat_tensor = np.stack(feat_mats, axis=0).transpose(1, 2, 0)  # [T, n_stocks, F]
    feature_dim = feat_tensor.shape[-1]

    scaler = StandardScaler()

    dataset = []
    corr_window = max(2, int(corr_window))
    corr_threshold = float(corr_threshold)

    first_error = None

    for i in range(seq_len, len(dates) - 2):
        try:
            # 1) node features: a single reshape (performance optimization)
            slab = feat_tensor[i - seq_len:i].astype(np.float32, copy=False)  # [T, A, F]
            node_feats = slab.reshape(-1, slab.shape[-1])  # [T*A, F]
            # rolling-window standardization
            start_idx = max(0, i - corr_window)  # rolling-window start
            rolling_window_feats = feat_tensor[start_idx:i].reshape(-1, slab.shape[-1])  # [W, A, F] -> [W*A, F]
            # mean/std of the current window
            rolling_mean = np.mean(rolling_window_feats, axis=0)
            rolling_std = np.std(rolling_window_feats, axis=0)
            # standardize the current node features
            x_np = (node_feats - rolling_mean) / (rolling_std + 1e-8)  # avoid division by zero
            x = torch.FloatTensor(x_np)  # [seq_len * n_stocks, F]

            # 2) temporal edges (bidirectional, w=0)
            edge_triplets = []  # (u, v, w)
            for a in range(n_stocks):
                for t in range(seq_len - 1):
                    cur = t * n_stocks + a
                    nxt = (t + 1) * n_stocks + a
                    edge_triplets.extend([(cur, nxt, 0.0), (nxt, cur, 0.0)])

            # 3) correlation edges
            recent = returns.iloc[i - corr_window:i].values  # [W, n_stocks]
            if recent.shape[0] >= 2:
                corr = np.corrcoef(recent.T)
                for j in range(n_stocks):
                    for k in range(j + 1, n_stocks):
                        w = float(corr[j, k])
                        if abs(w) > corr_threshold:
                            last_j = (seq_len - 1) * n_stocks + j
                            last_k = (seq_len - 1) * n_stocks + k
                            w = max(-0.999, min(0.999, w))
                            edge_triplets.extend([(last_j, last_k, w), (last_k, last_j, w)])

            if edge_triplets:
                us = [u for (u, v, w) in edge_triplets]
                vs = [v for (u, v, w) in edge_triplets]
                ws = [w for (u, v, w) in edge_triplets]
                edge_index = torch.LongTensor([us, vs])
                edge_attr = torch.FloatTensor(ws)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0,), dtype=torch.float32)

            # 4) supervision target: next day's cross-sectional return
            y = torch.FloatTensor(returns.iloc[i + 1].values).to(torch.float32)  # [n_stocks]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            dataset.append(data)

        except Exception as e:
            if first_error is None:
                import traceback
                first_error = f"Failed to build the first sample @i={i}: {e}\n{traceback.format_exc()}"
            continue

    if first_error and len(dataset) == 0:
        raise RuntimeError(first_error)

    print(f"Successfully created {len(dataset)} samples (node feature dim F={feature_dim}, assets={n_stocks})")

    preview = feature_cols[:8] if isinstance(feature_cols, list) else []
    print(f"Using raw feature columns: {preview}{' ...' if isinstance(feature_cols, list) and len(feature_cols)>8 else ''}")
    return dataset, feature_cols
