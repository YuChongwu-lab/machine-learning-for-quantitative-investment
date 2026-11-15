from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


#Adds CutMix to the training function
#CutMix data augmentation
@torch.no_grad()
def cutmix_tabular_columns(x, y, alpha=1.0, col_frac=(0.3, 0.7)):
    """
    Tabular CutMix: randomly pick a subset of columns (features) and
    replace them with the corresponding columns from another sample.
    - x: [B, F] float32
    - y: [B]    long
    - alpha: Beta distribution parameter (controls randomness of mix strength)
    - col_frac: (low, high) range for the replaced-column ratio, with the
      final ratio r ~ Uniform(low, high)

    Returns:
      x_mixed: [B, F]
      y_a: the original labels
      y_b: the mixed-in labels (after shuffling the pairing)
      lam: this batch's mixing weight scalar ~ 1 - r (the fraction of
           features kept from the original sample, used for the mixed loss)
    """
    B, F = x.size()
    # randomly shuffle the pairing
    perm = torch.randperm(B, device=x.device)
    x2, y2 = x[perm], y[perm]

    # randomly choose the replaced-column ratio r
    r = np.random.uniform(col_frac[0], col_frac[1])
    k = max(1, int(r * F))  # number of columns to replace

    # build the mask: 1 keeps the original sample's column, 0 replaces it with the paired sample's
    mask = torch.ones(F, device=x.device, dtype=x.dtype)
    idx = torch.randperm(F, device=x.device)[:k]
    mask[idx] = 0.0
    # broadcast across the batch: [B, F]
    mask = mask.unsqueeze(0).expand(B, -1)

    x_mixed = x * mask + x2 * (1.0 - mask)

    # approximate lam as "fraction of columns kept from the original sample"
    # (a single scalar shared across the batch, to avoid touching the loss computation logic)
    lam = float(mask.mean().item())  # ~= 1 - r
    return x_mixed, y, y2, lam

#Add CutMix inside the training function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience,
                use_cutmix=False, cutmix_prob=0.5, cutmix_alpha=1.0, cutmix_col_frac=(0.3, 0.7),
                checkpoint_path="best_model.pt"):
    best_val_loss = float('inf')
    best_state = None
    early_stop_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # ==== Optional: CutMix (column-wise replacement) ====
            if use_cutmix and np.random.rand() < cutmix_prob:
                x_mixed, y_a, y_b, lam = cutmix_tabular_columns(
                    x_batch, y_batch, alpha=cutmix_alpha, col_frac=cutmix_col_frac
                )
                outputs = model(x_mixed)
                # linear combination of the two CE losses (criterion stays mean-reduced, unchanged)
                loss = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)
            else:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
            # ================================

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1:02d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), checkpoint_path)
    return history


#Evaluate the model
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            total_loss += criterion(outputs, y).item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1

#Prediction function
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            all_preds.extend(model(x).argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
    return all_preds, all_labels

#Visualization
def plot_results(history, test_metrics, test_loader, model, device, output_dir="."):
    """Plot the loss curve, validation-metric curves, and confusion matrix.

    ``output_dir`` is where the figures are written. The original code
    hardcoded all three figures to the current directory, so experiments
    in this section overwrote each other -- which is exactly why
    pictures/ had manually-renamed images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.plot(history["train_loss"], label="Train Loss",linestyle='-')
    plt.plot(history["val_loss"], label="Val Loss",linestyle=':')
    plt.legend(); plt.grid(); plt.title("Loss Curve",fontsize=24); plt.xlabel("Epoch",fontsize=20);plt.ylabel("Loss",fontsize=20)
    plt.savefig(output_dir / "loss_curve.png"); plt.close()

    plt.plot(history["val_accuracy"], label="Val Accuracy",linestyle='-')
    plt.plot(history["val_f1"], label="Val F1 Score",linestyle=':')
    plt.legend(); plt.grid(); plt.title("Validation Metrics",fontsize=24); plt.xlabel("Epoch",fontsize=20)
    plt.savefig(output_dir / "val_metrics.png"); plt.close()

    print(f"\nTest Results - Loss: {test_metrics[0]:.4f}, Accuracy: {test_metrics[1]:.4f}, F1: {test_metrics[2]:.4f}")

    preds, labels = get_predictions(model, test_loader, device)
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix",fontsize=24)
    plt.xlabel("Predicted",fontsize=18); plt.ylabel("True",fontsize=18)
    plt.savefig(output_dir / "confusion_matrix.png"); plt.close()

