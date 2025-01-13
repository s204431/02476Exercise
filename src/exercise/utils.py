import matplotlib.pyplot as plt
import torch


def show_image_and_target(images: torch.Tensor, target: torch.Tensor, show: bool = True) -> None:
    """Show images and target labels."""
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.set_title(f"Label: {target[i]}")
        ax.axis("off")
    if show:
        plt.show()