"""Data loading and graph-dataset construction.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def load_real_data(data_file_path: str,
                   max_assets: int = 30,
                   max_days: int = 500) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load the raw data, returning price (pivoted price matrix), bench
    (equal-weight benchmark NAV), and panel_data (the raw panel, aligned/filtered to match)."""
    print(f"Loading data: {data_file_path}")
    if not isinstance(data_file_path, str):
        raise TypeError("data_file_path should be a string path")
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")

    # read the raw panel; pkl/csv/excel are supported
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

    if not pd.api.types.is_datetime64_any_dtype(panel_data['trade_date']):  # check the date dtype
        panel_data['trade_date'] = pd.to_datetime(panel_data['trade_date'])
    panel_data = panel_data.sort_values(['trade_date', 'ts_code'])

    # pivot the prices, fill, drop missing
    price = panel_data.pivot(index='trade_date', columns='ts_code', values='close')
    price = price.sort_index().ffill().bfill()  # forward-fill first (handles e.g. trading halts), then back-fill any remaining gaps (e.g. at the start of the series)
    price = price.dropna(axis=0, how='all').dropna(axis=1, how='all')  # drop dates where every stock is missing, and stocks with no data on any date

    # cap the scale (columns first, then rows)
    if max_assets is not None and max_assets > 0:
        price = price.iloc[:, :max_assets]
    if max_days is not None and max_days > 0:
        price = price.iloc[-max_days:]

    # build the benchmark NAV from equal-weight returns
    eq_ret = price.pct_change().mean(axis=1).fillna(0.0)
    bench = pd.Series(np.cumprod(1 + eq_ret), index=price.index, name='Benchmark')
    bench.iloc[0] = 1000  # make sure the benchmark NAV starts at 1000

    # align panel_data's dates and stock codes with price, for data consistency
    keep_dates = set(price.index)
    keep_codes = set(price.columns)
    mask = panel_data['trade_date'].isin(keep_dates) & panel_data['ts_code'].isin(keep_codes)
    panel_data = panel_data.loc[mask].copy()

    print(f"Processed price-data shape: {price.shape} (assets={price.shape[1]}, days={price.shape[0]})")
    return price, bench, panel_data


def _construct_time_edges(seq_len: int, n_stocks: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate bidirectional temporal edges for each asset's time series."""
    time_steps = torch.arange(seq_len - 1)
    asset_indices = torch.arange(n_stocks)

    # forward edges: t -> t+1
    src_forward = time_steps.repeat_interleave(n_stocks) * n_stocks + asset_indices.repeat(seq_len - 1)
    dst_forward = (time_steps + 1).repeat_interleave(n_stocks) * n_stocks + asset_indices.repeat(seq_len - 1)

    # backward edges: t+1 -> t
    src_backward = dst_forward
    dst_backward = src_forward

    # merge both directions
    edge_index = torch.cat([
        torch.stack([src_forward, dst_forward], dim=0),
        torch.stack([src_backward, dst_backward], dim=0)
    ], dim=1)

    # all temporal edges have weight 0
    edge_attr = torch.zeros(edge_index.size(1), dtype=torch.float32)

    return edge_index, edge_attr


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

    # separate numeric from non-numeric features
    all_feat_cols = feature_cols[:]
    numeric_cols = panel_data[all_feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = sorted(list(set(all_feat_cols) - set(numeric_cols)))

    # handle non-numeric features: optionally drop, or factorize (integer-encode)
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

    # convert the panel's feature data into a tensor suitable for the model
    feat_mats = []
    for col in feature_cols:
        mat = panel_data.pivot(index='trade_date', columns='ts_code', values=col)
        mat = mat.reindex(index=dates, columns=price.columns)
        mat = mat.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        feat_mats.append(mat.values)
    feat_tensor = np.stack(feat_mats, axis=0).transpose(1, 2, 0)  # [T, n_stocks, F]
    feature_dim = feat_tensor.shape[-1]

    # build the graph-structured data
    dataset = []
    corr_window = max(2, int(corr_window))  # correlation window
    corr_threshold = float(corr_threshold)  # correlation threshold

    first_error = None  # captures the first error encountered

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

            # 2) build the temporal edges (bidirectional, w=0)
            time_edge_index, time_edge_attr = _construct_time_edges(seq_len, n_stocks)

            # 3) build the correlation edges: a bidirectional edge where |corr| over
            # corr_window exceeds corr_threshold, weighted by the correlation
            recent = returns.iloc[i - corr_window:i].values  # [corr_window, n_stocks]

            if recent.shape[0] >= 2:
                # compute the correlation matrix
                corr = np.corrcoef(recent.T)  # [n_stocks, n_stocks]

                # get the upper-triangle indices (excluding the diagonal)
                triu_indices = np.triu_indices(n_stocks, k=1)
                j_indices = triu_indices[0]  # source-node indices
                k_indices = triu_indices[1]  # target-node indices

                # extract the upper-triangle correlation values
                corr_values = corr[j_indices, k_indices]  # [n_pairs]

                # find the edges that exceed the threshold
                mask = np.abs(corr_values) > corr_threshold
                valid_j = j_indices[mask]
                valid_k = k_indices[mask]
                valid_w = corr_values[mask]

                if len(valid_j) > 0:
                    # compute the node indices (at the last time step)
                    last_step_offset = (seq_len - 1) * n_stocks
                    src_nodes = last_step_offset + valid_j
                    dst_nodes = last_step_offset + valid_k

                    # clip the weight range to [-0.999, 0.999]
                    valid_w = np.clip(valid_w, -0.999, 0.999)

                    # build the bidirectional edges (vectorized concatenation)
                    corr_edge_src = np.concatenate([src_nodes, dst_nodes])
                    corr_edge_dst = np.concatenate([dst_nodes, src_nodes])
                    corr_edge_weights = np.concatenate([valid_w, valid_w])

                    # convert to tensors
                    corr_edge_index = torch.LongTensor(np.stack([corr_edge_src, corr_edge_dst]))
                    corr_edge_attr = torch.FloatTensor(corr_edge_weights)

                    # merge the temporal and correlation edges
                    edge_index = torch.cat([time_edge_index, corr_edge_index], dim=1)
                    edge_attr = torch.cat([time_edge_attr, corr_edge_attr], dim=0)
                else:
                    # no edges exceeded the threshold; use only the temporal edges
                    edge_index = time_edge_index
                    edge_attr = time_edge_attr
            else:
                # not enough data for the correlation window; use only the temporal edges
                edge_index = time_edge_index
                edge_attr = time_edge_attr
            # 4) supervision target: next day's cross-sectional return
            # the next day's cross-sectional return is the prediction target [n_stocks]
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
