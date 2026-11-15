import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# -------------------------- Data preprocessing --------------------------
def sort_features_by_correlation(data, label_col=-1, use_abs=True):
    df = pd.DataFrame(data)
    features = df.iloc[:,:label_col]
    labels = df.iloc[:, label_col]
    corrs = features.corrwith(labels)
    if use_abs:
        corrs = corrs.abs()
    sorted_idx = corrs.sort_values(ascending=False).index.tolist()
    center_sorted = []
    for i, idx in enumerate(sorted_idx):
        if i % 2 == 0:
            center_sorted.insert(0, idx)
        else:
            center_sorted.append(idx)
    sorted_features = features[center_sorted].values
    sorted_data = np.hstack([sorted_features, labels.values.reshape(-1, 1)])
    return sorted_data, center_sorted



def make_images(data, label_quantile, image_shape=(9, 11)):
    """
        Convert the data into 9x11 images, one per row, and factorize the
        label into quantile bins.
        """
    data = np.array(data)
    images = []
    labels = []

    for row in data:
        features = np.array(row[:-1], dtype=np.float32)  # features are the first 93 columns
        label = row[-1]  # label is the last column
        # pad the remaining feature slots with 0 (up to 99 elements)
        padded = np.pad(features, (0, image_shape[0] * image_shape[1] - len(features)), mode='constant', constant_values=0)
        # reshape into a 9x11 image
        image = padded.reshape(image_shape)
        images.append(image)
        labels.append(label)

    labels=pd.Series(labels)
    labels =labels.transform(lambda x: pd.qcut(x, label_quantile, labels=False, duplicates='drop'))
    images = np.array(images)
    images= images[:,np.newaxis,:,:]
    labels =  np.array(labels)
    return images.astype(np.float32), labels.astype(np.int64)# image data, shape [N, 1, 9, 11]; label data, shape [N,]


def create_dataloader(images, labels, batch_size=32, shuffle=True):
    dataset = TensorDataset(torch.tensor(images), torch.tensor(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)









