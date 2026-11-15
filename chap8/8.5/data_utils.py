"""Data loading, multi-edge-type graph construction, and feature engineering."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import torch

PANEL_DATA = None


def load_real_data(data_file_path: str, benchmark_file_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load stock data in panel format (pkl/csv/xlsx supported).

    Parameters:
    - data_file_path: path to the panel data file
    - benchmark_file_path: path to the benchmark data file (optional)

    Returns:
    - price_data: pivoted price-data table (DataFrame)
    - benchmark_data: benchmark data (Series)
    """
    print("Loading panel data...")

    # Load the data -- several formats are supported
    if data_file_path.endswith('.pkl'):
        panel_data = pd.read_pickle(data_file_path)
    elif data_file_path.endswith('.csv'):
        panel_data = pd.read_csv(data_file_path)
    elif data_file_path.endswith(('.xlsx', '.xls')):
        panel_data = pd.read_excel(data_file_path)
    else:
        raise ValueError("Supported formats: pkl, csv, xlsx, xls")

    print(f"Raw data shape: {panel_data.shape}")
    print(f"Raw data columns: {list(panel_data.columns)}")

    # Validate the data
    required_columns = ['trade_date', 'ts_code', 'ret', 'close']
    missing_columns = [col for col in required_columns if col not in panel_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert the date column
    if not pd.api.types.is_datetime64_any_dtype(panel_data['trade_date']):
        panel_data['trade_date'] = pd.to_datetime(panel_data['trade_date'])

    # Sort by date and stock code
    panel_data = panel_data.sort_values(['trade_date', 'ts_code'])

    # Clean the data
    print("Cleaning data...")

    # Compute each stock's missing-value ratio
    stock_missing_stats = panel_data.groupby('ts_code')['close'].apply(lambda x: x.isnull().mean())
    valid_stocks = stock_missing_stats[stock_missing_stats < 0.1].index
    panel_data = panel_data[panel_data['ts_code'].isin(valid_stocks)]

    # Remove outliers
    panel_data = panel_data[(panel_data['close'] > 0) & (panel_data['close'] < 1e6)]
    panel_data = panel_data[(panel_data['ret'] > -0.5) & (panel_data['ret'] < 0.5)]

    # Remove stocks with fewer than 100 data points
    stock_counts = panel_data.groupby('ts_code').size()
    valid_stocks = stock_counts[stock_counts >= 100].index
    panel_data = panel_data[panel_data['ts_code'].isin(valid_stocks)]

    print(f"Cleaned data shape: {panel_data.shape}")
    print(f"Number of valid stocks: {panel_data['ts_code'].nunique()}")

    # Build the price-data pivot table
    print("Building the price matrix...")
    price_data = panel_data.pivot(index='trade_date', columns='ts_code', values='close')

    # Forward/backward-fill missing values
    price_data = price_data.ffill().bfill()
    price_data = price_data.dropna(axis=0, how='all').dropna(axis=1, how='all')

    # Make sure there are enough stocks and a long enough time series
    if price_data.shape[1] < 10:
        raise ValueError(f"Not enough valid stocks: {price_data.shape[1]}, need at least 10")
    if price_data.shape[0] < 100:
        raise ValueError(f"Time series too short: {price_data.shape[0]}, need at least 100 days")

    print(f"Price-data shape: {price_data.shape}")
    print(f"Date range: {price_data.index[0]} to {price_data.index[-1]}")
    print(f"Number of stocks: {len(price_data.columns)}")
    print(f"Example stock codes: {list(price_data.columns[:5])}")

    # Build the benchmark data
    if benchmark_file_path:
        print("Loading benchmark data...")
        if benchmark_file_path.endswith('.pkl'):
            benchmark_data = pd.read_pickle(benchmark_file_path)
            if isinstance(benchmark_data, pd.DataFrame):
                benchmark_data = benchmark_data.iloc[:, 0]
        elif benchmark_file_path.endswith('.csv'):
            benchmark_data = pd.read_csv(benchmark_file_path, index_col=0, parse_dates=True).squeeze()
        else:
            benchmark_data = pd.read_excel(benchmark_file_path, index_col=0, parse_dates=True).squeeze()

        # Align the two time series
        common_dates = price_data.index.intersection(benchmark_data.index)
        if len(common_dates) > 0:
            price_data = price_data.loc[common_dates]
            benchmark_data = benchmark_data.loc[common_dates]
        else:
            print("Benchmark dates don't overlap with the price data; using an equal-weight benchmark instead")
            benchmark_data = None

    if benchmark_file_path is None or benchmark_data is None:
        print("Generating an equal-weight benchmark...")
        equal_weight_returns = price_data.pct_change().mean(axis=1)
        benchmark_prices = [1000]
        for ret in equal_weight_returns[1:]:
            benchmark_prices.append(benchmark_prices[-1] * (1 + ret))
        benchmark_data = pd.Series(benchmark_prices, index=price_data.index, name='Benchmark')

    # Stash the panel data for later use
    global PANEL_DATA
    PANEL_DATA = panel_data

    return price_data, benchmark_data


class MultiEdgeGraphConstructor:
    """A graph constructor supporting multiple, independently selectable edge types."""

    def __init__(self, edge_types: Set[str] = None, correlation_threshold: float = 0.4, knn_k: int = 8):
        if edge_types is None:
            edge_types = {'correlation', 'sector', 'knn', 'temporal'}
        self.edge_types = edge_types
        self.correlation_threshold = correlation_threshold
        self.knn_k = knn_k
        print(f"The graph constructor will build the following edge types: {self.edge_types}")

    def construct_correlation_graph(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Correlation edges: a bidirectional edge is created between any pair whose
        |correlation| exceeds the threshold, weighted by the correlation coefficient."""
        if 'correlation' not in self.edge_types:
            return np.array([]), np.array([]), False  # empty arrays + a failure flag

        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            corr_matrix = np.corrcoef(returns.T)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        except Exception:
            return np.array([]), np.array([]), False

        np.fill_diagonal(corr_matrix, 0)
        edge_list = []
        edge_weights = []

        for i in range(corr_matrix.shape[0]):
            for j in range(i + 1, corr_matrix.shape[1]):
                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    edge_list.extend([(i, j), (j, i)])
                    edge_weights.extend([corr_matrix[i, j], corr_matrix[i, j]])

        if len(edge_list) == 0:
            return np.array([]), np.array([]), False

        edge_index = np.array(edge_list).T
        edge_attr = np.array(edge_weights)
        return edge_index, edge_attr, True

    def construct_sector_graph(self, sector_mapping: Dict[str, str], stock_list: List[str]) -> Tuple[
        np.ndarray, np.ndarray, bool]:
        """Sector edges: a bidirectional edge is created between every pair of stocks in the
        same sector (no cross-sector edges), weighted uniformly at 1."""
        if 'sector' not in self.edge_types:
            return np.array([]), np.array([]), False

        edge_list = []
        edge_weights = []
        stock_to_idx = {stock: idx for idx, stock in enumerate(stock_list)}

        # Group by sector
        sector_groups = {}
        for stock in stock_list:
            if stock in sector_mapping:
                sector = sector_mapping[stock]
                if sector not in sector_groups:
                    sector_groups[sector] = []
                sector_groups[sector].append(stock_to_idx[stock])

        # Connect every pair of stocks within the same sector
        for sector, stock_indices in sector_groups.items():
            if len(stock_indices) > 1:
                for i in range(len(stock_indices)):
                    for j in range(i + 1, len(stock_indices)):
                        idx_i, idx_j = stock_indices[i], stock_indices[j]
                        edge_list.extend([(idx_i, idx_j), (idx_j, idx_i)])
                        edge_weights.extend([1.0, 1.0])

        if len(edge_list) == 0:
            return np.array([]), np.array([]), False

        edge_index = np.array(edge_list).T
        edge_attr = np.array(edge_weights)
        return edge_index, edge_attr, True

    def construct_knn_graph(self, features: np.ndarray, k: int = None) -> Tuple[np.ndarray, np.ndarray, bool]:
        """KNN edges: a bidirectional edge is created to each of the k nearest neighbors by
        cosine similarity, provided the similarity exceeds 0.1, weighted by the similarity."""
        if 'knn' not in self.edge_types:
            return np.array([]), np.array([]), False

        if k is None:
            k = self.knn_k

        from sklearn.metrics.pairwise import cosine_similarity

        n_samples = features.shape[0]
        k = min(k, n_samples - 1)

        try:
            similarity_matrix = cosine_similarity(features)
            np.fill_diagonal(similarity_matrix, -1)
        except Exception:
            return np.array([]), np.array([]), False

        edge_list = []
        edge_weights = []

        for i in range(n_samples):
            neighbors = np.argsort(similarity_matrix[i])[-k:]
            similarities = similarity_matrix[i][neighbors]

            for j, sim in zip(neighbors, similarities):
                if sim > 0.1:
                    edge_list.extend([(i, j), (j, i)])
                    edge_weights.extend([sim, sim])

        if len(edge_list) == 0:
            return np.array([]), np.array([]), False

        edge_index = np.array(edge_list).T
        edge_attr = np.array(edge_weights)
        return edge_index, edge_attr, True

    def construct_temporal_edges(self, n_assets: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Temporal edges: a bidirectional edge is created between a stock's consecutive
        time steps, weighted uniformly at 1."""
        if 'temporal' not in self.edge_types:
            return np.array([]), np.array([]), False

        edge_list = []
        edge_weights = []

        for asset in range(n_assets):
            for t in range(seq_len - 1):
                current_node = t * n_assets + asset
                next_node = (t + 1) * n_assets + asset
                edge_list.extend([(current_node, next_node), (next_node, current_node)])
                edge_weights.extend([1.0, 1.0])

        edge_index = np.array(edge_list).T
        edge_attr = np.array(edge_weights)
        return edge_index, edge_attr, True


class FeatureEngineer:
    """Feature engineering supporting multiple edge types."""

    def __init__(self):
        self.scaler = StandardScaler()

    def extract_technical_features(self, price_series: pd.Series) -> pd.DataFrame:
        """Extract technical-indicator features for a single asset."""
        features = pd.DataFrame(index=price_series.index)

        # Basic features
        features['returns'] = price_series.pct_change().fillna(0)
        features['log_returns'] = np.log(price_series / price_series.shift(1)).fillna(0)

        # Moving averages
        for window in [5, 10, 20]:
            ma = price_series.rolling(window, min_periods=1).mean()
            features[f'ma_{window}'] = ma
            features[f'price_ma_ratio_{window}'] = price_series / ma.where(ma != 0, 1)

        # Technical indicators
        features['volatility_5'] = features['returns'].rolling(5, min_periods=1).std().fillna(0.01)
        features['volatility_20'] = features['returns'].rolling(20, min_periods=1).std().fillna(0.01)

        # Momentum features
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = price_series.pct_change(period).fillna(0)

        # RSI
        delta = features['returns']
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.where(loss != 0, 1e-8)
        features['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = price_series.ewm(span=12, min_periods=1).mean()
        ema_26 = price_series.ewm(span=26, min_periods=1).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9, min_periods=1).mean()

        # Bollinger bands
        rolling_mean = price_series.rolling(20, min_periods=1).mean()
        rolling_std = price_series.rolling(20, min_periods=1).std()
        features['bollinger_upper'] = rolling_mean + (rolling_std * 2)
        features['bollinger_lower'] = rolling_mean - (rolling_std * 2)
        bb_range = features['bollinger_upper'] - features['bollinger_lower']
        features['bollinger_position'] = (price_series - features['bollinger_lower']) / bb_range.where(bb_range != 0, 1)

        # Clean up outliers
        features = features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        for col in features.columns:
            features[col] = np.clip(features[col], -1e6, 1e6)

        return features

    def extract_panel_features(self, panel_data: pd.DataFrame, price_data: pd.DataFrame) -> List[pd.DataFrame]:
        """Extract features from the panel data."""
        print("Extracting features from the panel data...")

        # Get all feature columns (besides the base columns), de-duplicated
        base_columns = ['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol', 'ret', 'industry']
        feature_columns = list(set([col for col in panel_data.columns if col not in base_columns]))

        print(f"Found extra feature columns: {feature_columns}")

        all_features_list = []

        for stock in price_data.columns:
            stock_panel_data = panel_data[panel_data['ts_code'] == stock].set_index('trade_date').sort_index()

            # Align to the common time index
            stock_panel_data = stock_panel_data.reindex(price_data.index).ffill()

            # Extract basic price features
            stock_price = price_data[stock]
            tech_features = pd.DataFrame(index=price_data.index)

            # Basic features
            tech_features['returns'] = stock_price.pct_change().fillna(0)
            tech_features['log_returns'] = np.log(stock_price / stock_price.shift(1)).fillna(0)

            # Price-relative features
            for window in [5, 10, 20]:
                ma = stock_price.rolling(window, min_periods=1).mean()
                tech_features[f'price_ma_ratio_{window}'] = stock_price / ma.where(ma != 0, 1)

            # Reuse the technical indicators already present in the panel data
            panel_features = pd.DataFrame(index=price_data.index)

            # Use the precomputed indicators directly
            if 'momentum_14' in stock_panel_data.columns:
                panel_features['momentum_14'] = stock_panel_data['momentum_14']
            if 'volatility_14' in stock_panel_data.columns:
                panel_features['volatility_14'] = stock_panel_data['volatility_14']
            if 'RSI_14' in stock_panel_data.columns:
                panel_features['RSI_14'] = stock_panel_data['RSI_14']

            # Basic market-data features
            for col in ['open', 'high', 'low', 'vol']:
                if col in stock_panel_data.columns:
                    panel_features[f'{col}'] = stock_panel_data[col]
                    if col == 'vol':
                        panel_features[f'{col}_ma_ratio'] = stock_panel_data[col] / stock_panel_data[col].rolling(20,
                                                                                                                  min_periods=1).mean()

            # Return features
            if 'ret' in stock_panel_data.columns:
                panel_features['panel_ret'] = stock_panel_data['ret']
                # Lagged return features
                for lag in [1, 5]:
                    panel_features[f'ret_lag{lag}'] = stock_panel_data['ret'].shift(lag)

            # Combine all features
            combined_features = pd.concat([tech_features, panel_features], axis=1)

            # Clean the data
            combined_features = combined_features.replace([np.inf, -np.inf], np.nan)
            combined_features = combined_features.ffill().fillna(0)

            # Clip the value range
            numeric_cols = combined_features.select_dtypes(include=[np.number]).columns
            combined_features[numeric_cols] = combined_features[numeric_cols].clip(-1e6, 1e6)

            all_features_list.append(combined_features)

        return all_features_list

    def create_dataset_with_configurable_edges(self, price_data: pd.DataFrame,
                                               panel_data: pd.DataFrame,
                                               seq_len: int = 20,
                                               edge_types: Set[str] = None,
                                               max_stocks: int = 190,
                                               lookback_days: int = 2500,
                                               sample_interval: int = 5,
                                               correlation_threshold: float = 0.4,
                                               knn_k: int = 8) -> List[Data]:
        """Build a graph dataset with configurable edge types -- fixed version (avoids
        data leakage, makes sure the edge-type selection actually takes effect)."""
        if edge_types is None:
            edge_types = {'correlation', 'sector', 'knn', 'temporal'}

        print("Building a configurable-edge-type graph dataset from the panel data (rolling normalization)...")
        print(f"Enabled edge types: {edge_types}")

        # ===== 1) Cap the scale and align =====
        print("Reducing the data size...")
        stock_counts = panel_data.groupby('ts_code').size().sort_values(ascending=False)
        top_stocks = stock_counts.head(max_stocks).index

        latest_date = price_data.index.max()
        start_date = latest_date - pd.Timedelta(days=lookback_days)

        price_data_limited = price_data.loc[start_date:, top_stocks].dropna(axis=1, how='all')
        panel_data_limited = panel_data[
            (panel_data['ts_code'].isin(price_data_limited.columns)) &
            (panel_data['trade_date'] >= start_date)
            ].copy()

        self.price_data_used = price_data_limited

        # ===== 2) Extract the sector mapping (optional) =====
        sector_mapping = {}
        if 'industry' in panel_data_limited.columns:
            sector_info = panel_data_limited.groupby('ts_code')['industry'].first()
            sector_mapping = sector_info.to_dict()
            print(f"Found sector information, covering {len(set(sector_mapping.values()))} sectors")

        print(f"Reduced price-data shape: {price_data_limited.shape}")
        print(f"Reduced panel-data shape: {panel_data_limited.shape}")

        # ===== 3) Extract the (unnormalized) feature matrix =====
        all_tech_features = self.extract_panel_features(panel_data_limited, price_data_limited)

        print("Reassembling the feature matrix...")
        # Unify the feature-column set (take the first min_features columns common to all assets)
        min_features = min(len(f.select_dtypes(include=[np.number]).columns) for f in all_tech_features)
        reference_features = all_tech_features[0].select_dtypes(include=[np.number])
        common_feature_names = reference_features.columns[:min_features]

        normalized_features = []
        for asset_features in all_tech_features:
            numeric_features = asset_features.select_dtypes(include=[np.number])
            aligned_features = numeric_features.reindex(columns=common_feature_names, fill_value=0)
            normalized_features.append(aligned_features)

        n_features_per_asset = len(common_feature_names)
        feature_names = []
        for asset_idx, asset in enumerate(price_data_limited.columns):
            for feature in common_feature_names:
                feature_names.append(f"{asset}_{feature}")

        # Assemble the "time x (asset*feature)" table
        feature_array = np.zeros((len(normalized_features[0]), len(feature_names)))
        for asset_idx, asset_features in enumerate(normalized_features):
            start_col = asset_idx * n_features_per_asset
            end_col = start_col + n_features_per_asset
            feature_array[:, start_col:end_col] = asset_features.values

        tech_features_combined = pd.DataFrame(
            feature_array,
            index=normalized_features[0].index,
            columns=feature_names
        )

        # Basic cleanup (but no global normalization here)
        tech_features_combined = tech_features_combined.replace([np.inf, -np.inf], np.nan) \
            .ffill().fillna(0)
        tech_features_combined = tech_features_combined.clip(-1e6, 1e6)

        # ===== 4) Graph constructor and sector edges (precomputed once) =====
        print("Constructing the configurable multi-type graph data...")
        graph_constructor = MultiEdgeGraphConstructor(edge_types=edge_types,
                                                        correlation_threshold=correlation_threshold,
                                                        knn_k=knn_k)
        dataset = []
        n_assets = len(price_data_limited.columns)
        stock_list = list(price_data_limited.columns)

        # Precompute the sector edges (only once)
        sector_edge_index, sector_edge_attr, sector_success = graph_constructor.construct_sector_graph(sector_mapping,
                                                                                                       stock_list)
        if sector_success:
            print(f"Successfully built the sector edges: {len(sector_edge_attr)} edges")
        else:
            print("No sector edges built (not enabled, or no data)")

        # ===== 5) Generate samples (the key to rolling normalization: fit_transform separately per window) =====
        total_samples = (len(tech_features_combined) - seq_len - 1) // sample_interval
        print(f"About to generate ~{total_samples} samples...")

        for idx, i in enumerate(range(seq_len, len(tech_features_combined) - 1, sample_interval)):
            if idx % 20 == 0:
                print(f"Progress: {idx}/{total_samples}")

            try:
                # 5.1 Take this sample's time window
                window_data = tech_features_combined.iloc[i - seq_len:i]

                # 5.2 Rolling normalization: fit and transform using only this window's data
                scaler = StandardScaler()
                window_scaled = pd.DataFrame(
                    scaler.fit_transform(window_data),
                    index=window_data.index,
                    columns=window_data.columns
                )

                # 5.3 Build the node features [seq_len * n_assets, n_features_per_asset]
                node_features = []
                for t in range(seq_len):
                    for asset_idx in range(n_assets):
                        start_col = asset_idx * n_features_per_asset
                        end_col = start_col + n_features_per_asset
                        asset_features = window_scaled.iloc[t, start_col:end_col].values
                        node_features.append(asset_features)

                x = torch.FloatTensor(np.array(node_features))
                if torch.isnan(x).any() or torch.isinf(x).any():
                    continue

                # ===== 6) Edge-construction logic =====
                all_edges, all_edge_attrs, all_edge_types = [], [], []
                edge_stats = {'correlation': 0, 'sector': 0, 'knn': 0, 'temporal': 0}

                # 6.1 Correlation edges (type=0): based on the past seq_len days of returns,
                # connected only at the last time layer
                recent_returns = price_data_limited.iloc[i - seq_len:i].pct_change().dropna()
                corr_edge_index, corr_edge_attr, corr_success = graph_constructor.construct_correlation_graph(
                    recent_returns.values)
                if corr_success and corr_edge_index.shape[1] > 0:
                    last_time_corr_edges = corr_edge_index + (seq_len - 1) * n_assets
                    all_edges.append(last_time_corr_edges)
                    all_edge_attrs.extend(corr_edge_attr)
                    all_edge_types.extend([0] * len(corr_edge_attr))
                    edge_stats['correlation'] = len(corr_edge_attr)

                # 6.2 Sector edges (type=1): duplicated at every time step
                if sector_success:
                    for t in range(seq_len):
                        time_sector_edges = sector_edge_index + t * n_assets
                        all_edges.append(time_sector_edges)
                        all_edge_attrs.extend(sector_edge_attr)
                        all_edge_types.extend([1] * len(sector_edge_attr))
                    edge_stats['sector'] = len(sector_edge_attr) * seq_len

                # 6.3 KNN edges (type=2): based on the last time step's normalized features
                last_time_features = []
                for asset_idx in range(n_assets):
                    start_col = asset_idx * n_features_per_asset
                    end_col = start_col + n_features_per_asset
                    last_time_features.append(window_scaled.iloc[-1, start_col:end_col].values)
                last_time_features = np.array(last_time_features)

                knn_edge_index, knn_edge_attr, knn_success = graph_constructor.construct_knn_graph(last_time_features)
                if knn_success and knn_edge_index.shape[1] > 0:
                    last_time_knn_edges = knn_edge_index + (seq_len - 1) * n_assets
                    all_edges.append(last_time_knn_edges)
                    all_edge_attrs.extend(knn_edge_attr)
                    all_edge_types.extend([2] * len(knn_edge_attr))
                    edge_stats['knn'] = len(knn_edge_attr)

                # 6.4 Temporal edges (type=3): connect adjacent time layers within the window
                temporal_edge_index, temporal_edge_attr, temporal_success = graph_constructor.construct_temporal_edges(
                    n_assets, seq_len)
                if temporal_success and temporal_edge_index.shape[1] > 0:
                    all_edges.append(temporal_edge_index)
                    all_edge_attrs.extend(temporal_edge_attr)
                    all_edge_types.extend([3] * len(temporal_edge_attr))
                    edge_stats['temporal'] = len(temporal_edge_attr)

                # ===== Handle the no-edges case =====
                if len(all_edges) == 0:
                    print(f"Warning: sample {idx} has no edges at all, skipping")
                    continue

                # Print edge statistics (every 20 samples)
                if idx % 20 == 0:
                    print(f"Sample {idx} edge stats: {edge_stats}")

                final_edge_index = np.concatenate(all_edges, axis=1)
                final_edge_attr = np.array(all_edge_attrs)
                final_edge_types = np.array(all_edge_types)

                # ===== 7) Target label: each stock's next-day return =====
                next_date = price_data_limited.index[i + 1]
                next_returns = []
                for stock in price_data_limited.columns:
                    stock_next_data = panel_data_limited[
                        (panel_data_limited['ts_code'] == stock) &
                        (panel_data_limited['trade_date'] == next_date)
                        ]
                    if len(stock_next_data) > 0 and 'ret' in stock_next_data.columns:
                        ret = stock_next_data['ret'].iloc[0]
                        next_returns.append(ret if not np.isnan(ret) else 0)
                    else:
                        next_returns.append(0)
                next_returns = np.nan_to_num(np.array(next_returns), nan=0.0)

                # ===== 8) Build the Data object =====
                data = Data(
                    x=x,
                    edge_index=torch.LongTensor(final_edge_index),
                    edge_attr=torch.FloatTensor(final_edge_attr),
                    edge_type=torch.LongTensor(final_edge_types),
                    y=torch.FloatTensor(next_returns)
                )
                dataset.append(data)

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

        # ===== 9) Summary statistics =====
        print(f"Successfully generated {len(dataset)} samples")
        if len(dataset) > 0:
            avg_edges = np.mean([data.edge_index.shape[1] for data in dataset])
            print(f"Average edges per sample: {avg_edges:.1f}")

            edge_type_stats = {}
            for data in dataset[:5]:
                if hasattr(data, 'edge_type') and len(data.edge_type) > 0:
                    types, counts = torch.unique(data.edge_type, return_counts=True)
                    for t, c in zip(types.tolist(), counts.tolist()):
                        edge_type_name = ['correlation', 'sector', 'knn', 'temporal'][t]
                        edge_type_stats.setdefault(edge_type_name, []).append(c)

            print("Edge-type statistics (averaged over the first 5 samples):")
            for edge_type_name, counts in edge_type_stats.items():
                if counts:
                    print(f"  {edge_type_name}: {np.mean(counts):.1f}")

            # Verify the edge-type selection actually took effect
            print("\nEdge-type configuration check:")
            for edge_type in ['correlation', 'sector', 'knn', 'temporal']:
                if edge_type in edge_types:
                    type_idx = ['correlation', 'sector', 'knn', 'temporal'].index(edge_type)
                    has_this_type = any(type_idx in data.edge_type.tolist() for data in dataset[:5])
                    print(f"  {edge_type} edges: {'yes, enabled and built' if has_this_type else 'enabled but NOT built'}")
                else:
                    print(f"  {edge_type} edges: - not enabled")

        return dataset
