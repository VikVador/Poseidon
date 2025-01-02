r"""Poseidon - Tools to perform the training of a denoiser."""

import matplotlib.pyplot as plt

# isort: split


def plot_images_from_tensor(tensor, index):
    """
    Given a 4D tensor of shape [6, 33, 128, 128], this function takes an index for the first dimension
    and creates a subplot displaying the 33 images in a grid format.

    Parameters:
        tensor (torch.Tensor): The input tensor of shape [6, 33, 128, 128]
        index (int): The index to slice the first dimension (batch dimension)
    """
    # Select the images for the given index
    images = tensor[:, index]

    # Create the figure with a grid of 33 subplots (you can adjust the grid size as needed)
    fig, axes = plt.subplots(
        6, 6, figsize=(15, 15)
    )  # 6x6 grid for 33 images, adjust figsize as needed

    for i in range(33):
        ax = axes[i // 6, i % 6]
        ax.imshow(images[i].cpu().numpy(), cmap="viridis", vmin=-3, vmax=3)
        ax.axis("off")
    plt.tight_layout()
    return fig
