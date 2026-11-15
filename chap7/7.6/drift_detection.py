from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from paths import stock_file


def detect_distribution_shift(data, window_size=100):
    """Sliding-window KL-divergence drift detection."""
    shifts = []
    for i in range(len(data) - window_size * 2 + 1):
        window1 = data[i:i + window_size]
        window2 = data[i + window_size:i + 2 * window_size]

        hist1 = np.histogram(window1, bins=20, density=True)[0] + 1e-10
        hist2 = np.histogram(window2, bins=20, density=True)[0] + 1e-10

        kl_div = stats.entropy(hist1, hist2)
        shifts.append(kl_div)
    return np.array(shifts)


def visualize_shift_detection(data, window_size=100):
    """Visualize drift detection."""
    kl_scores = detect_distribution_shift(data, window_size)

    # Dynamic threshold (rolling mean + 3x rolling std)
    kl_series = pd.Series(kl_scores)
    rolling_mean = kl_series.rolling(window=50, min_periods=1).mean()
    rolling_std = kl_series.rolling(window=50, min_periods=1).std()
    dynamic_threshold = rolling_mean + 3 * rolling_std

    # Handle the initial NaNs
    global_mean = kl_scores.mean()
    global_std = kl_scores.std()
    dynamic_threshold = dynamic_threshold.fillna(global_mean + 3 * global_std)

    # Detect drift points
    shift_indices = np.where(kl_scores > dynamic_threshold)[0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'hspace': 0.4})

    # Raw data plot
    axes[0].plot(data.index, data.values, alpha=0.7, label='Raw data')
    axes[0].set_title('Raw Data Drift Point', fontsize=12)
    axes[0].set_ylabel('Daily Return(%)')

    axes[0].xaxis.set_major_locator(plt.MaxNLocator(10))
    for label in axes[0].get_xticklabels():
        label.set_rotation(30)
    for idx in shift_indices:
        pos = idx + window_size
        axes[0].axvline(x=data.index[pos], color='r', linestyle='--', alpha=0.5,
                        label='Drift Point' if idx == shift_indices[0] else "")
    axes[0].legend()

    # Second plot: KL divergence (x-axis is date)
    kl_index = data.index[window_size: window_size + len(kl_scores)]
    axes[1].plot(kl_index, kl_scores, label='KL Divergence')
    axes[1].plot(kl_index, dynamic_threshold, 'r--', label='Dynamic Threshold (rolling mean + 3σ)')
    axes[1].set_title('KL Divergence with Dynamic Threshold', fontsize=12)
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(10))
    for label in axes[1].get_xticklabels():
        label.set_rotation(30)
    axes[1].legend()

    # Third plot: distribution comparison
    if len(shift_indices) > 0:
        shift_pos = shift_indices[0]
        drift_index = shift_pos + window_size
        drift_date = data.index[drift_index]

        window1 = data[shift_pos:shift_pos + window_size]
        window2 = data[shift_pos + window_size:shift_pos + 2 * window_size]

        # Pre-drift: white histogram
        bars1 = sns.histplot(window1, bins=20, kde=True, ax=axes[2],
                             label='Pre-drift', alpha=0.5, color='white', edgecolor='black')

        # Make the pre-drift KDE line a black dashed line
        if axes[2].lines:
            axes[2].lines[0].set_color('black')
            axes[2].lines[0].set_linestyle('--')

        # Post-drift: gray histogram
        bars2 = sns.histplot(window2, bins=20, kde=True, ax=axes[2],
                             label='Post-drift', alpha=0.5, color='gray', edgecolor='black')

        axes[2].set_title(f'Distribution Comparison (Drift at {drift_date})')
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, 'No significant drift detected', ha='center', va='center')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "outputs" / "drift_detection"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stock in ("688256.SH", "600111.SH", "000538.SZ"):
        data = pd.read_csv(stock_file(f'{stock}.csv'), index_col='trade_date')
        fig = visualize_shift_detection(data.iloc[:, -1], window_size=100)
        fig.savefig(out_dir / f"{stock}_drift_detection.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_dir / f'{stock}_drift_detection.png'}")
