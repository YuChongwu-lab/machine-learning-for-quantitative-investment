
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.manifold import TSNE
from pyts.image import MarkovTransitionField
from sklearn.model_selection import train_test_split


#------------Generate pseudo-structured image data-------------------------------
def make_images(data, label_quantile, method="raw", image_shape=None,
                random_state=42, mtf_bins=8):
    """
    Convert tabular data into images; the last column is the label, the rest are features.
    Args:
        data: np.ndarray or pd.DataFrame, shape [N, D+1]
        label_quantile: int, number of label quantile bins
        method: {"raw","deepinsight","mtf","igtd"}
        image_shape: (H,W), only used by raw; auto square if not given
        random_state: t-SNE random seed
        mtf_bins: number of MTF discretization bins
    Returns:
        images: np.ndarray, shape depends on method
            raw/deepinsight/igtd -> [N,1,H,W]
            mtf -> [N,1,D,D]
        labels: np.ndarray, [N,]
    """
    data = np.asarray(data)
    features = data[:, :-1].astype(np.float32)  # features are already normalized, no further scaling here
    labels = data[:, -1]

    # Factorize the label into quantile bins (kept)
    labels = pd.Series(labels).transform(
        lambda x: pd.qcut(x, label_quantile, labels=False, duplicates='drop')
    ).to_numpy(dtype=np.int64)

    N, D = features.shape

    # -------- 1) Raw: pad + reshape directly (baseline) --------
    if method == "raw":
        if image_shape is None:
            h = w = int(np.ceil(np.sqrt(D)))
            image_shape = (h, w)
        H, W = image_shape
        padded = np.pad(features, ((0, 0), (0, H * W - D)), mode="constant", constant_values=0)
        images = padded.reshape(N, 1, H, W).astype(np.float32)
        return images, labels

    # Shared helper: map 2D coordinates onto a grid and aggregate collisions (mean)
    def rasterize(coords_2d, H, W):
        # coords_2d: [D,2], normalize to [0,1] first
        minv = coords_2d.min(axis=0)
        maxv = coords_2d.max(axis=0)
        scale = np.where((maxv - minv) > 1e-9, (maxv - minv), 1.0)
        coords01 = (coords_2d - minv) / scale

        rr = np.clip(np.round(coords01[:, 0] * (H - 1)).astype(int), 0, H - 1)
        cc = np.clip(np.round(coords01[:, 1] * (W - 1)).astype(int), 0, W - 1)

        # build pixel -> feature-index buckets
        buckets = defaultdict(list)
        for i, (r, c) in enumerate(zip(rr, cc)):
            buckets[(r, c)].append(i)

        img = np.zeros((N, H, W), dtype=np.float32)
        # for each pixel position, average the features mapped there (per sample)
        for (r, c), idxs in buckets.items():
            vals = features[:, idxs]                 # [N, K]
            img[:, r, c] = vals.mean(axis=1) if vals.ndim == 2 else vals
        return img

    # -------- 2) DeepInsight: features -> t-SNE -> square grid --------
    if method == "deepinsight":
        # Run t-SNE over the "feature dimension" (treat each feature as a D-dimensional point)
        tsne = TSNE(n_components=2, random_state=random_state, init="random", learning_rate="auto")
        coords = tsne.fit_transform(features.T)  # [D,2]

        H = W = int(np.ceil(np.sqrt(D)))
        img = rasterize(coords, H, W)            # [N,H,W]
        images = img[:, np.newaxis, :, :]        # [N,1,H,W]
        return images, labels

    # -------- 3) MTF: transition field directly on the length-D sequence --------
    if method == "mtf":
        mtf = MarkovTransitionField(image_size=D, n_bins=mtf_bins)
        img = mtf.fit_transform(features)        # [N,D,D]
        images = img[:, np.newaxis, :, :].astype(np.float32)
        return images, labels

    # -------- 4) IGTD: layout based on the correlation matrix (t-SNE on corr rows) --------
    if method == "igtd":
        corr = np.corrcoef(features.T)           # [D,D]
        # Reduce the "correlation matrix's row vectors" with t-SNE for a stable layout
        tsne = TSNE(n_components=2, random_state=random_state, init="random", learning_rate="auto")
        coords = tsne.fit_transform(corr)        # [D,2]

        H = W = int(np.ceil(np.sqrt(D)))
        img = rasterize(coords, H, W)            # [N,H,W]
        images = img[:, np.newaxis, :, :].astype(np.float32)
        return images, labels


    raise ValueError(f"Unknown method: {method}")


#-------------------Create the image-data dataloader----------------------------------------
def create_dataloader(images, labels, batch_size, shuffle=True):
    dataset = TensorDataset(torch.tensor(images), torch.tensor(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)




#-----------------------Split the raw data into train/val/test---------------------
def split_raw(data, label_quantile=None, test_size=0.2, val_size=0.1, random_state=42):

    if hasattr(data, "values"):  # convert DataFrame to numpy if needed
        data = data.values

    X = data[:, :-1].astype("float32")   # the first 93 columns are features
    y = data[:, -1].astype("float32")      # the last column is the label (classification task)
    #Factorize the label
    if label_quantile is not None and label_quantile > 1:
        # bin into quantiles with pandas.qcut
        y = pd.qcut(y, q=label_quantile, labels=False, duplicates="drop").astype("int64")
    else:
        y = y.astype("int64")  # already discrete classes

    # first split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # then split val out of train+val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=random_state, stratify=y_trainval
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

#-------------------Create a dataloader from the raw data
def create_dataloader_raw(X, y, batch_size=64, shuffle=True):
    """
    Convert a numpy array into a PyTorch DataLoader.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)  # long dtype for classification

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader







