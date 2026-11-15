"""Visualize the effect of tabular-to-image encoding (a standalone side experiment for this section).

Shows the first image in the dataset under IGTD encoding. Change
``method`` to compare the other encoding methods:
``raw`` / ``deepinsight`` / ``mtf`` / ``igtd``.

    python visualize_images.py
"""

import matplotlib.pyplot as plt

from data_utils import make_images
from get_data import get_data

#: Image encoding method; one of raw / deepinsight / mtf / igtd
METHOD = "igtd"

def visualize_first_image(images, labels, cmap="gray"):
    """
    Visualize the first image in the dataset.
    Args:
        images: np.ndarray, shape [N, 1, H, W]
        labels: np.ndarray, shape [N,]
        cmap: str, matplotlib colormap, defaults to grayscale
    """

    if len(images.shape) != 4:
        raise ValueError(f"input images should have shape [N, 1, H, W], got {images.shape}")

    first_image = images[0, 0, :, :]  # take the first image (H, W)
    first_label = labels[0]  # its corresponding label

    plt.figure(figsize=(4, 4))
    plt.imshow(first_image, cmap=cmap)
    plt.title(f"First Image (Label={first_label})", fontsize=12)
    plt.axis("off")
    plt.show()

def main():
    data = get_data()
    images, labels = make_images(data, label_quantile=2, method=METHOD,
                                 image_shape=None, random_state=42)
    visualize_first_image(images, labels)


if __name__ == "__main__":
    main()
