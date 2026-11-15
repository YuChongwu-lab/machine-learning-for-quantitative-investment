"""Visualization and debugging helpers for the multi-edge-type graph dataset and backtest."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from torch_geometric.data import Data


def visualize_edge_types(dataset: List[Data], price_data: pd.DataFrame,
                         sector_mapping: Optional[Dict[str, str]] = None,
                         sample_idx: int = 0):
    """
    Visualize the structure of the four edge types.
    """
    if len(dataset) <= sample_idx:
        print(f"Sample index out of range, max index: {len(dataset) - 1}")
        return

    sample_data = dataset[sample_idx]
    stock_names = list(price_data.columns)
    n_assets = len(stock_names)
    seq_len = sample_data.x.shape[0] // n_assets

    print(f"Debug info: n_assets={n_assets}, seq_len={seq_len}, total_nodes={sample_data.x.shape[0]}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Four Edge-Type Structures (sample {sample_idx})', fontsize=16)

    # Split out the edge types
    edge_index = sample_data.edge_index.numpy()
    edge_types = sample_data.edge_type.numpy()

    type_names = ['Correlation Edges', 'Sector Edges', 'KNN Edges', 'Temporal Edges']
    colors = ['red', 'blue', 'green', 'orange']

    print(f"Edge-type counts: {np.bincount(edge_types)}")

    for edge_type in range(4):
        ax = axes[edge_type // 2, edge_type % 2]

        # Filter to this edge type
        mask = edge_types == edge_type
        if not mask.any():
            ax.text(0.5, 0.5, f'No {type_names[edge_type]}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(type_names[edge_type])
            continue

        current_edges = edge_index[:, mask]
        print(f"{type_names[edge_type]}: {current_edges.shape[1]} edges")

        if edge_type == 3:  # temporal edges
            # Build a space-time graph, capped for display
            max_display_assets = min(8, n_assets)
            max_display_time = min(10, seq_len)

            G = nx.DiGraph()  # a directed graph makes the temporal relationship clearer

            # Add nodes
            node_positions = {}
            for t in range(max_display_time):
                for s in range(max_display_assets):
                    node_id = t * n_assets + s
                    if node_id < sample_data.x.shape[0]:  # make sure the node exists
                        G.add_node(node_id, time=t, stock=s)
                        node_positions[node_id] = (t * 2, s)  # x axis is time, y axis is stock

            # Add the temporal edges
            edge_count = 0
            for i in range(current_edges.shape[1]):
                source, target = current_edges[:, i]

                # Check whether the nodes fall within the display range
                source_t = source // n_assets
                source_s = source % n_assets
                target_t = target // n_assets
                target_s = target % n_assets

                if (source_t < max_display_time and source_s < max_display_assets and
                        target_t < max_display_time and target_s < max_display_assets and
                        source in G.nodes() and target in G.nodes()):
                    G.add_edge(source, target)
                    edge_count += 1

            print(f"Temporal graph display: {len(G.nodes())} nodes, {edge_count} edges")

            if len(G.nodes()) > 0:
                # Draw the temporal graph
                nx.draw(G, node_positions, ax=ax, node_size=100,
                        node_color=colors[edge_type], edge_color=colors[edge_type],
                        alpha=0.7, with_labels=True, font_size=6, arrows=True)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Stock Index')
            else:
                ax.text(0.5, 0.5, 'Error processing the temporal-edge data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)

        elif edge_type == 2:  # KNN edges
            # Extract the last time step's KNN edges
            last_time_start = (seq_len - 1) * n_assets
            last_time_end = seq_len * n_assets

            G = nx.Graph()
            max_display_stocks = min(15, n_assets)
            G.add_nodes_from(range(max_display_stocks))

            knn_edge_count = 0
            for i in range(current_edges.shape[1]):
                source, target = current_edges[:, i]

                # Check whether this is an edge from the last time step
                if last_time_start <= source < last_time_end and last_time_start <= target < last_time_end:
                    s_source = source - last_time_start
                    s_target = target - last_time_start

                    if s_source < max_display_stocks and s_target < max_display_stocks:
                        G.add_edge(s_source, s_target)
                        knn_edge_count += 1

            print(f"KNN graph display: {len(G.nodes())} nodes, {knn_edge_count} edges")

            if len(G.edges()) > 0:
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, ax=ax, node_color=colors[edge_type], node_size=200,
                        edge_color=colors[edge_type], alpha=0.7, with_labels=True,
                        font_size=8)
            else:
                ax.text(0.5, 0.5, 'KNN edges: no connections, or a data-processing error', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)

        else:  # correlation and sector edges
            last_time_start = (seq_len - 1) * n_assets
            last_time_end = seq_len * n_assets

            last_time_edges = []
            for i in range(current_edges.shape[1]):
                source, target = current_edges[:, i]
                if last_time_start <= source < last_time_end and last_time_start <= target < last_time_end:
                    s_source = source - last_time_start
                    s_target = target - last_time_start
                    if s_source < n_assets and s_target < n_assets:
                        last_time_edges.append((s_source, s_target))

            if last_time_edges:
                G = nx.Graph()
                max_display_stocks = min(15, n_assets)
                G.add_nodes_from(range(max_display_stocks))

                for source, target in last_time_edges:
                    if source < max_display_stocks and target < max_display_stocks:
                        G.add_edge(source, target)

                pos = nx.circular_layout(G)

                if edge_type == 1 and sector_mapping:  # sector edges -- color by sector
                    node_colors = []
                    sector_color_map = {}
                    color_palette = plt.cm.Set3(np.linspace(0, 1, 12))

                    for i in range(max_display_stocks):
                        if i < len(stock_names):
                            stock = stock_names[i]
                            sector = sector_mapping.get(stock, 'Unknown')
                            if sector not in sector_color_map:
                                sector_color_map[sector] = color_palette[len(sector_color_map) % 12]
                            node_colors.append(sector_color_map[sector])
                        else:
                            node_colors.append('gray')
                else:
                    node_colors = colors[edge_type]

                nx.draw(G, pos, ax=ax, node_color=node_colors, node_size=200,
                        edge_color=colors[edge_type], alpha=0.7, with_labels=True,
                        font_size=8)

        ax.set_title(f'{type_names[edge_type]} ({np.sum(mask)} edges)')

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nEdge-type statistics:")
    for i, name in enumerate(type_names):
        count = np.sum(edge_types == i)
        print(f"{name}: {count} edges")


def plot_temporal_edge_dynamics(dataset: List[Data], n_samples: int = 5):
    """
    Plot how the number of temporal edges changes over the dataset.
    """
    if len(dataset) < n_samples:
        n_samples = len(dataset)

    fig, ax = plt.subplots(figsize=(12, 8))

    temporal_edge_counts = []
    sample_indices = []

    for i in range(0, len(dataset), max(1, len(dataset) // n_samples)):
        sample_data = dataset[i]
        edge_types = sample_data.edge_type.numpy()
        temporal_count = np.sum(edge_types == 3)  # temporal edges are type 3
        temporal_edge_counts.append(temporal_count)
        sample_indices.append(i)

    ax.plot(sample_indices, temporal_edge_counts, 'o-', color='orange', linewidth=2, markersize=8)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Number of Temporal Edges')
    ax.set_title('Number of Temporal Edges over the Dataset')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_backtest_results(backtest_results: Dict, portfolio_values: List[float],
                          benchmark_data: pd.Series, dates: pd.DatetimeIndex,
                          optimization_method: str = 'softmax_long_only'):
    """Plot the backtest results."""
    try:
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Hierarchical Gated Dynamic GNN Backtest Results ({optimization_method})', fontsize=16, fontweight='bold')

        # Net-value curve
        ax1 = axes[0, 0]
        portfolio_returns = np.array(portfolio_values) / portfolio_values[0]

        if len(benchmark_data) >= len(portfolio_returns):
            benchmark_start_idx = len(benchmark_data) - len(portfolio_returns)
            benchmark_returns = benchmark_data.iloc[benchmark_start_idx:] / benchmark_data.iloc[benchmark_start_idx]
        else:
            benchmark_returns = benchmark_data / benchmark_data.iloc[0]
            portfolio_returns = portfolio_returns[-len(benchmark_data):]
            dates = dates[-len(benchmark_data):]

        ax1.plot(dates[-len(portfolio_returns):], portfolio_returns, label=f'{optimization_method} strategy', linewidth=2,
                 color='blue')
        ax1.plot(dates[-len(portfolio_returns):], benchmark_returns, label='Benchmark Index', linewidth=2, color='red',
                 alpha=0.7)
        ax1.set_title('Cumulative Return Comparison')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # Drawdown analysis
        ax2 = axes[0, 1]
        cumulative_returns = portfolio_returns
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max

        ax2.fill_between(dates[-len(portfolio_returns):], drawdowns, 0, alpha=0.3, color='red')
        ax2.plot(dates[-len(portfolio_returns):], drawdowns, color='red', linewidth=1)
        ax2.set_title(f'Drawdown Analysis (Max Drawdown: {backtest_results["max_drawdown"]:.2%})')
        ax2.set_ylabel('Drawdown')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # Return distribution
        ax3 = axes[1, 0]
        daily_returns = np.diff(portfolio_returns) / portfolio_returns[:-1]
        ax3.hist(daily_returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(np.mean(daily_returns), color='red', linestyle='--', label=f'Mean: {np.mean(daily_returns):.4f}')
        ax3.set_title('Daily Return Distribution')
        ax3.set_xlabel('Daily Return')
        ax3.set_ylabel('Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Key metrics
        ax4 = axes[1, 1]
        ax4.axis('off')

        metrics_text = f"""
{optimization_method.upper()} Strategy Performance:

Total Return: {backtest_results['total_return']:.2%}
Annualized Return: {backtest_results['annual_return']:.2%}
Annualized Volatility: {backtest_results['annual_volatility']:.2%}
Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}
Max Drawdown: {backtest_results['max_drawdown']:.2%}
Win Rate: {backtest_results['win_rate']:.2%}
Final Value: ${backtest_results['final_portfolio_value']:,.2f}
        """

        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("matplotlib is not installed; skipping the plots")
        print(f"\n=== {optimization_method.upper()} Strategy Backtest Results ===")
        for key, value in backtest_results.items():
            if isinstance(value, float):
                if 'rate' in key or 'return' in key:
                    print(f"{key}: {value:.2%}")
                elif 'ratio' in key:
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")


def debug_edge_construction(dataset: List[Data], sample_idx: int = 0):
    """Debug the edge-construction process."""
    if len(dataset) <= sample_idx:
        print("Sample index out of range")
        return

    sample = dataset[sample_idx]
    print(f"Debug info for sample {sample_idx}:")
    print(f"Number of nodes: {sample.x.shape[0]}")
    print(f"Node feature dimension: {sample.x.shape[1]}")
    print(f"Number of edges: {sample.edge_index.shape[1]}")

    if hasattr(sample, 'edge_type'):
        edge_types = sample.edge_type.numpy()
        type_names = ['Correlation Edges', 'Sector Edges', 'KNN Edges', 'Temporal Edges']

        print("\nEdge-type distribution:")
        for i, name in enumerate(type_names):
            count = np.sum(edge_types == i)
            print(f"  {name}: {count}")

            if count > 0:
                mask = edge_types == i
                edges = sample.edge_index[:, mask].numpy()
                print(f"    Node range: {edges.min()} - {edges.max()}")
                print(f"    First 5 edges: {edges[:, :min(5, edges.shape[1])].T}")

    print(f"\nTarget value range: {sample.y.min():.6f} - {sample.y.max():.6f}")
