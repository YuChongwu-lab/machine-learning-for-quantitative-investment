
"""
Concept representation and distribution-drift detection.
Functionality:
1. extract the concept representation, check for outliers, descriptive statistics
2. compute concept drift, descriptive statistics
3. compute adjacent-concept similarity (cosine similarity), descriptive statistics
4. visualization: the concept-vector distribution, delta norm over time, delta norm distribution, adjacent-concept similarity, per-dimension variance, PCA visualization

"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from TCVAE import TCVAEConceptEncoder

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """Run the full concept-representation and distribution-drift analysis, and produce the visualization."""
    print("=" * 80)
    print("Concept representation and distribution-drift detection")
    print("=" * 80)
    print()

    # ------------------- Base config -------------------
    import copy
    from config import BASE_CFG
    CFG = copy.deepcopy(BASE_CFG)
    CFG["max_assets"] = 1  # this analysis script only uses 1 stock

    # =================== 1. Load the data ===================
    print("[1/7] Loading data...")
    from data import load_real_data, make_windows_xy

    try:
        price, bench, panel_data = load_real_data(
            CFG["data_path"],
            max_assets=2,
            max_days=1000
        )
        print(f"   Data loaded successfully")

        stock_id = panel_data['ts_code'].unique()[0]
        df_one = panel_data[panel_data['ts_code'] == stock_id].sort_values('trade_date')
        print(f"  Test stock: {stock_id}")
        print(f"  Data length: {len(df_one)} days")
    except Exception as e:
        print(f"  Data loading failed: {e}")
        sys.exit(1)

    # =================== 2. Feature engineering ===================
    print("\n[2/7] Feature engineering...")
    seq_len = CFG["seq_len"]
    pred_len = CFG["pred_len"]


    feature_cols = [c for c in df_one.columns if c not in ["trade_date", "ts_code", "close"]]
    cont_cols = [c for c in feature_cols if df_one[c].nunique() > 20]#treated as continuous if it has more than 20 distinct values
    disc_cols = [c for c in feature_cols if c not in cont_cols]#everything else is treated as discrete

    X_cont=df_one[cont_cols]
    # Discrete features: check for and filter out constants
    if len(disc_cols) > 0:
        disc_data = []
        for col in disc_cols:
            if df_one[col].nunique() > 1:
                disc_data.append(df_one[col].fillna(0).values.reshape(-1, 1))
        X_disc = np.concatenate(disc_data, axis=1) if disc_data else np.zeros((len(df_one), 0))
    else:
        X_disc = np.zeros((len(df_one), 0))

    X_all = np.concatenate([X_cont, X_disc], axis=1)
    y_full = np.log(df_one["close"]).diff().shift(-1).fillna(0).values

    # Train/test split
    n = len(X_all)
    if n < (seq_len + pred_len + 5):
        raise ValueError("Not enough data to build any windows.")
    n_train = int(n * 0.8)

    # Standardize
    mu = X_all[:n_train].mean(axis=0, keepdims=True)
    std = X_all[:n_train].std(axis=0, keepdims=True) + 1e-6
    X_all = (X_all - mu) / std

    X_train_all = X_all[:n_train]
    y_train_all = y_full[:n_train]
    X_test_all = X_all[n_train:]
    y_test_all = y_full[n_train:]

    # Build time windows
    Xw_tr, Yw_tr = make_windows_xy(X_train_all, y_train_all, seq_len, pred_len)
    Xw_test, Yw_test = make_windows_xy(X_test_all, y_test_all, seq_len, pred_len)


    feature_dim = Xw_tr.shape[-1]

    print(f"Data preparation complete")
    print(f"Training set: {len(Xw_tr)} samples")
    print(f"Test set: {len(Xw_test)} samples")
    print(f"Feature dimension: {feature_dim}")


    # =================== 3. Build the encoder ===================
    print("\n[3/7] Building the concept encoder...")

    latent_dim = 64
    hidden_dim = 128

    encoder = TCVAEConceptEncoder(
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        pooling='attention'
    ).to(DEVICE)

    print(f"Feature dimension: {feature_dim}")
    print(f"Concept dimension: {latent_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Parameter count: {sum(p.numel() for p in encoder.parameters()):,}")

    # =================== 4. Extract concept vectors ===================
    print("\n[4/7] Extracting concept vectors...")

    encoder.eval()
    with torch.no_grad():
        # Convert all windows into a tensor
        Xw_tensor = torch.tensor(Xw_tr, dtype=torch.float32).to(DEVICE)

        # Extract the concept vectors
        concepts = encoder(Xw_tensor).cpu().numpy()  # (N, latent_dim)

        print(f"Concept-vector shape: {concepts.shape}")

    # Descriptive statistics
    print(f"\n  Concept-vector statistics:")
    print(f"Mean: {concepts.mean():.6f}")
    print(f"Std: {concepts.std():.6f}")
    print(f"Min: {concepts.min():.6f}")
    print(f"Max: {concepts.max():.6f}")
    print(f"Median: {np.median(concepts):.6f}")

    # Check for outliers
    zero_ratio = (np.abs(concepts) < 1e-6).mean()
    inf_ratio = np.isinf(concepts).mean()
    nan_ratio = np.isnan(concepts).mean()

    print(f"\n  Outlier check:")
    print(f"Near-zero fraction: {zero_ratio:.2%}")
    print(f"Inf fraction: {inf_ratio:.2%}")
    print(f"NaN fraction: {nan_ratio:.2%}")

    # =================== 5. Compute concept drift ===================
    print("\n[5/7] Computing concept drift (delta)...")

    # Simulate Proceed's rolling anchor
    window_size = 10  # anchor window size
    deltas = []
    delta_norms = []

    for i in range(window_size, len(concepts)):
        # current concept
        c_t = concepts[i]
        # anchor: mean of the previous window_size concepts
        c_anchor = concepts[i - window_size:i].mean(axis=0)
        # drift
        delta = c_t - c_anchor
        deltas.append(delta)
        delta_norms.append(np.linalg.norm(delta))

    deltas = np.array(deltas)
    delta_norms = np.array(delta_norms)

    print(f" Computed {len(deltas)} drift vectors")
    print(f"\n  Concept-drift statistics:")
    print(f" Delta mean: {deltas.mean():.6f}")
    print(f" Delta std: {deltas.std():.6f}")
    print(f" Delta-norm mean: {delta_norms.mean():.6f}")
    print(f" Delta-norm std: {delta_norms.std():.6f}")
    print(f" Delta-norm max: {delta_norms.max():.6f}")

    # =================== 6. Concept-vector discriminability ===================
    print("\n[6/7] Analyzing concept-vector discriminability...")

    # Compute the cosine similarity between adjacent concepts
    similarities = []
    for i in range(len(concepts) - 1):
        sim = cosine_similarity(concepts[i:i + 1], concepts[i + 1:i + 2])[0, 0]
        similarities.append(sim)

    similarities = np.array(similarities)

    print(f" Adjacent-concept similarity:")
    print(f" Mean: {similarities.mean():.6f}")
    print(f" Std: {similarities.std():.6f}")
    print(f"  Min: {similarities.min():.6f}")
    print(f"  Max: {similarities.max():.6f}")
    print(f"  Median: {np.median(similarities):.6f}")

    # Compute per-dimension variance of the concept vectors
    per_dim_var = concepts.var(axis=0)
    print(f"\n  Per-dimension variance:")
    print(f"    Mean: {per_dim_var.mean():.4f}")
    print(f"    Min: {per_dim_var.min():.4f}")
    print(f"    Max: {per_dim_var.max():.4f}")

    # Check whether any dimension has too little variance (inactive)
    inactive_dims = (per_dim_var < 0.001).sum()
    if inactive_dims > 0:
        print(f"Warning: {inactive_dims}/{latent_dim} dimensions have variance < 0.001 (possibly inactive)")
    else:
        print(f" All dimensions are active")

    # =================== 7. Visualization ===================
    print("\n[7/7] Generating visualization...")

    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Concept Representation and Distribution Drift Detection - {stock_id}', fontsize=16, fontweight='bold')

        # Plot 1: concept-vector distribution (dims 1 and 2)
        ax = axes[0, 0]
        ax.scatter(concepts[:, 0], concepts[:, 1], c=np.arange(len(concepts)),
                   cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel('Concept Dim 1')
        ax.set_ylabel('Concept Dim 2')
        ax.set_title('Concept Space Distribution (First 2 Dims)')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Time Order')

        # Plot 2: delta-norm time series
        ax = axes[0, 1]
        ax.plot(delta_norms, linewidth=1, alpha=0.7)
        ax.axhline(delta_norms.mean(), color='r', linestyle='--',
                   label=f'Mean={delta_norms.mean():.3f}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Delta Norm')
        ax.set_title('Concept Drift Magnitude')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: delta-norm distribution
        ax = axes[0, 2]
        ax.hist(delta_norms, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(delta_norms.mean(), color='r', linestyle='--',
                   label=f'Mean={delta_norms.mean():.3f}')
        ax.set_xlabel('Delta Norm')
        ax.set_ylabel('Count')
        ax.set_title('Concept Drift Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: adjacent-concept similarity
        ax = axes[1, 0]
        ax.plot(similarities, linewidth=1, alpha=0.7)
        ax.axhline(similarities.mean(), color='r', linestyle='--',
                   label=f'Mean={similarities.mean():.3f}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Adjacent-Concept Similarity')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 5: per-dimension variance
        ax = axes[1, 1]
        ax.bar(range(len(per_dim_var)), per_dim_var, alpha=0.7)
        ax.axhline(0.001, color='r', linestyle='--', label='Threshold=0.001')
        ax.set_xlabel('Concept Dimension')
        ax.set_ylabel('Variance')
        ax.set_title(f'Per-Dimension Variance ({inactive_dims} Low-Variance Dims)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 6: PCA visualization
        ax = axes[1, 2]
        if len(concepts) > 2:
            pca = PCA(n_components=2)
            concepts_2d = pca.fit_transform(concepts)
            scatter = ax.scatter(concepts_2d[:, 0], concepts_2d[:, 1],
                                 c=np.arange(len(concepts)), cmap='viridis',
                                 alpha=0.6, s=20)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_title('PCA Visualization')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Time Order')

        plt.tight_layout()

        # Save the figure
        save_path = 'checkpoints/concept_encoder_verification.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Visualization saved: {save_path}")

        # Show the figure if running in a notebook environment
        try:
            plt.show()
        except:
            pass

    except Exception as e:
        print(f"  Visualization failed: {e}")




if __name__ == "__main__":
    main()
