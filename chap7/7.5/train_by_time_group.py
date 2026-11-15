import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_utils import create_dataloader_raw, split_raw
from model.SAINT import SAINT
from paths import data_file
from train_utils import evaluate_model, train_model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed = 42
set_seed(seed)

# ============Train by time group ============
def train_by_time_groups(
        raw_data_path=data_file('data_ml.csv'),
        time_col='date',
        num_classes=2,
        batch_size=256,
        lr=1e-3,
        epochs=100,
        patience=10,
        test_size=0.2,
        val_size=0.16,
        device=None,
        sample_time_groups=None,
        sample_seed=42,
        use_cutmix=True,
        cutmix_prob=0.7,
        cutmix_alpha=1.0,
        cutmix_col_frac=(0.3, 0.7),
        **model_kwargs
):
    # Load the full data (keep stock_id as the index)
    full_data = pd.read_csv(raw_data_path, index_col='stock_id')

    # Reset the index so grouping can use positional indices
    full_data_reset = full_data.reset_index()  # turn stock_id into a regular column
    # Group by time
    groups = full_data_reset.groupby(time_col).groups
    time_points = sorted(groups.keys())


    # Sampling logic
    if sample_time_groups is not None and sample_time_groups < len(time_points):
        np.random.seed(sample_seed)
        sampled_indices = np.random.choice(len(time_points), sample_time_groups, replace=False)
        time_points = [time_points[i] for i in sorted(sampled_indices)]
        print(f"Randomly sampled {sample_time_groups} time group(s): {time_points}")



    all_results = []
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for t_idx, time_point in enumerate(time_points):
        print(f"\n{'=' * 60}")
        print(f"Processing Time Group [{t_idx + 1}/{len(time_points)}]: {time_point}")
        print(f"{'=' * 60}")

        # ===== Key fix 2: extract data by position with iloc =====
        group_row_indices = groups[time_point]  # list of integer positional indices
        group_data_with_time = full_data_reset.iloc[group_row_indices].copy()

        # Drop the time column and stock_id column (now a regular column after reset_index)
        group_data = group_data_with_time.drop(columns=[time_col, 'stock_id'])

        print(f"Raw sample count (including missing): {len(group_data)}")

        # Drop missing values
        group_data_clean = group_data.dropna()

        print(f"  Valid sample count (no missing values): {len(group_data_clean)}")
        print(f"   Number of feature+label columns: {group_data_clean.shape[1]}")

        if len(group_data_clean) < 100:
            print(f"    Insufficient sample size ({len(group_data_clean)}), skipping this group")
            continue

        # Split the dataset
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = split_raw(
                group_data_clean.values,
                label_quantile=num_classes,
                test_size=test_size,
                val_size=val_size,
                random_state=42
            )

            print(f"   Split result - train:{len(X_train)}, val:{len(X_val)}, test:{len(X_test)}")

        except ValueError as e:
            print(f"   Data split failed: {e}, skipping this group")
            continue

        # Create the DataLoaders
        train_loader = create_dataloader_raw(X_train, y_train, batch_size=batch_size)
        val_loader = create_dataloader_raw(X_val, y_val, batch_size=batch_size, shuffle=False)
        test_loader = create_dataloader_raw(X_test, y_test, batch_size=batch_size, shuffle=False)

        # Initialize the model
        num_features = X_train.shape[1]
        model = SAINT(
            num_features=num_features,
            num_classes=num_classes,
            **model_kwargs
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Train the model
        history = train_model(
            model, train_loader, val_loader, criterion, optimizer, device,
            epochs, patience,
            use_cutmix=use_cutmix, cutmix_prob=cutmix_prob, cutmix_alpha=cutmix_alpha,
            cutmix_col_frac=cutmix_col_frac
        )

        # Evaluate on the test set
        test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, criterion, device)

        # Record the results
        all_results.append({
            'time': time_point,
            'n_samples': len(group_data_clean),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'best_epoch': len(history['train_loss'])
        })

        print(f" Test set results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

        # Save this group's model
        torch.save(model.state_dict(), f"best_model_time_{time_point.replace('/', '-')}.pt")

    # Aggregate the results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("time_group_results.csv", index=False)

    print(f"\n{'=' * 60}")
    print("Overall statistics:")
    print(f"{'=' * 60}")
    if len(results_df) > 0:
        print(results_df[['time', 'n_samples', 'test_accuracy', 'test_f1']])

    return results_df


# ============ Main program ============
if __name__ == "__main__":
    # Parameters
    num_classes = 2
    lr = 1e-3
    test_size = 0.2
    val_size = 0.16
    patience = 10
    epochs = 100
    batch_size = 64
    num_features = 93
    d_model = 128
    n_heads = 8
    num_layers = 6
    dropout = 0.1

    # Model config
    model_config = {
        'd_model': d_model,
        'n_heads': n_heads,
        'n_blocks': num_layers,
        'dropout': dropout
    }

    # Train by time group
    results = train_by_time_groups(
        raw_data_path=data_file('data_ml.csv'),
        time_col='date',
        num_classes=num_classes,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        patience=patience,
        test_size=test_size,
        val_size=val_size,
        sample_time_groups=2,  # randomly sample 2 time groups
        sample_seed=42,  # random seed (for reproducibility)
        **model_config
    )

    # Visualize the overall results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(results['time'], results['test_loss'], marker='o')
    axes[0].set_title('Test Loss by Time', fontsize=16)
    axes[0].set_xlabel('Time')
    axes[0].grid(True)

    axes[1].plot(results['time'], results['test_accuracy'], marker='s', color='green')
    axes[1].set_title('Test Accuracy by Time', fontsize=16)
    axes[1].set_xlabel('Time')
    axes[1].grid(True)

    axes[2].plot(results['time'], results['test_f1'], marker='^', color='orange')
    axes[2].set_title('Test F1 Score by Time', fontsize=16)
    axes[2].set_xlabel('Time')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('time_series_metrics.png', dpi=150)
    plt.close()

    print("\n  All time groups finished training! Results saved to time_group_results.csv")
