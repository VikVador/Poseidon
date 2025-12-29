r"""Tools for spatial coarsening of ocean data."""

import torch
import torch.nn.functional as F
import xarray as xr

from torch import Tensor
from typing import Tuple

# isort: split
from poseidon.config import PATH_MASK_B
from poseidon.data.const import DATASET_REGION


def create_coarsen_variable(
    input_tensor: Tensor,
    lon_src: Tensor,
    lat_src: Tensor,
    lon_tgt: Tensor,
    lat_tgt: Tensor,
    target_resolution: Tuple[int, int],
    align_corners: bool = True,
) -> Tensor:
    r"""Coarsen a variable using bilinear interpolation.

    Arguments:
        input_tensor: Input tensor of shape (H_src, W_src).
        lon_src: Source longitude coordinates of shape (W_src,).
        lat_src: Source latitude coordinates of shape (H_src,).
        lon_tgt: Target longitude coordinates of shape (W_tgt,).
        lat_tgt: Target latitude coordinates of shape (H_tgt,).
        target_resolution: Target resolution (H_tgt, W_tgt).
        align_corners: Whether to align corners in grid_sample.

    Returns:
        Coarsened tensor of shape (H_tgt, W_tgt).
    """
    H_tgt, W_tgt = target_resolution
    H_src, W_src = input_tensor.shape

    # Convert to tensors on same device (detach to avoid forward AD issues)
    device = input_tensor.device
    dtype = input_tensor.dtype

    lon_src = torch.as_tensor(lon_src, device=device, dtype=dtype).detach()
    lat_src = torch.as_tensor(lat_src, device=device, dtype=dtype).detach()
    lon_tgt = torch.as_tensor(lon_tgt, device=device, dtype=dtype).detach()
    lat_tgt = torch.as_tensor(lat_tgt, device=device, dtype=dtype).detach()

    # Validate dimensions
    assert len(lat_src) == H_src, f"lat_src length ({len(lat_src)}) must match input height ({H_src})"
    assert len(lon_src) == W_src, f"lon_src length ({len(lon_src)}) must match input width ({W_src})"
    assert len(lat_tgt) == H_tgt, f"lat_tgt length ({len(lat_tgt)}) must match H_tgt ({H_tgt})"
    assert len(lon_tgt) == W_tgt, f"lon_tgt length ({len(lon_tgt)}) must match W_tgt ({W_tgt})"

    # Create coordinate grids
    lat_src_grid, lon_src_grid = torch.meshgrid(lat_src, lon_src, indexing="ij")
    lat_tgt_grid, lon_tgt_grid = torch.meshgrid(lat_tgt, lon_tgt, indexing="ij")

    # Normalize target coordinates to [-1, 1] for grid_sample
    x_tgt_norm = 2 * (lon_tgt_grid - lon_src_grid.min()) / (lon_src_grid.max() - lon_src_grid.min()) - 1
    y_tgt_norm = 2 * (lat_tgt_grid - lat_src_grid.min()) / (lat_src_grid.max() - lat_src_grid.min()) - 1

    # Create sampling grid: (1, H_tgt, W_tgt, 2) with [lon, lat] order (detached)
    grid = torch.stack([x_tgt_norm, y_tgt_norm], dim=-1).unsqueeze(0).detach()

    # Apply bilinear interpolation
    input_4d = input_tensor.unsqueeze(0).unsqueeze(0)
    output_4d = F.grid_sample(
        input_4d,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=align_corners,
    )

    return output_4d.squeeze(0).squeeze(0)


def create_coarsened_mask(
    target_resolution: Tuple[int, int],
    lon_tgt: Tensor,
    lat_tgt: Tensor,
    min_ocean_fraction: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> xr.DataArray:
    r"""Create a coarsened land-sea mask from the original high-resolution mesh.

    Arguments:
        target_resolution: Target resolution (H_tgt, W_tgt).
        lon_tgt: Target longitude coordinates.
        lat_tgt: Target latitude coordinates.
        min_ocean_fraction: Threshold for ocean classification (0-1).
        device: Device for tensor operations.

    Returns:
        Coarsened mask with shape (32, H_tgt, W_tgt) containing 0 (land) or 1 (ocean).
    """
    # Load original high-resolution mesh
    ds_mesh = xr.open_zarr(PATH_MASK_B).isel(**DATASET_REGION).load()
    mask_src = torch.tensor(ds_mesh["mask"].values, device=device, dtype=torch.float32)

    # Get source coordinates
    lon_src = torch.tensor(ds_mesh.longitude.values, device=device)
    lat_src = torch.tensor(ds_mesh.latitude.values, device=device)
    levels = ds_mesh.level.values

    # Convert target coordinates to tensors
    lon_tgt = torch.as_tensor(lon_tgt, device=device)
    lat_tgt = torch.as_tensor(lat_tgt, device=device)

    H_tgt, W_tgt = target_resolution

    # Validate dimensions
    assert len(lat_tgt) == H_tgt, f"lat_tgt length ({len(lat_tgt)}) must match H_tgt ({H_tgt})"
    assert len(lon_tgt) == W_tgt, f"lon_tgt length ({len(lon_tgt)}) must match W_tgt ({W_tgt})"

    # Create coordinate grids
    lat_src_grid, lon_src_grid = torch.meshgrid(lat_src, lon_src, indexing="ij")
    lat_tgt_grid, lon_tgt_grid = torch.meshgrid(lat_tgt, lon_tgt, indexing="ij")

    # Normalize target coordinates to [-1, 1]
    x_tgt_norm = 2 * (lon_tgt_grid - lon_src_grid.min()) / (lon_src_grid.max() - lon_src_grid.min()) - 1
    y_tgt_norm = 2 * (lat_tgt_grid - lat_src_grid.min()) / (lat_src_grid.max() - lat_src_grid.min()) - 1

    # Create sampling grid
    grid = torch.stack([x_tgt_norm, y_tgt_norm], dim=-1).unsqueeze(0)

    # Initialize output mask
    mask_tgt = torch.zeros((mask_src.shape[0], H_tgt, W_tgt), device=device)

    # Process each depth level
    for level_idx in range(mask_src.shape[0]):
        mask_level = mask_src[level_idx]

        # Interpolate to get ocean fraction
        ocean_frac = F.grid_sample(
            mask_level.unsqueeze(0).unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze()

        # Apply threshold
        mask_tgt[level_idx] = (ocean_frac > min_ocean_fraction).float()

    # Convert to xarray DataArray
    mask_coarsened = xr.DataArray(
        mask_tgt.cpu().numpy(),
        dims=["level", "latitude", "longitude"],
        coords={
            "level": levels,
            "latitude": lat_tgt.cpu().numpy() if isinstance(lat_tgt, torch.Tensor) else lat_tgt,
            "longitude": lon_tgt.cpu().numpy() if isinstance(lon_tgt, torch.Tensor) else lon_tgt,
        },
        attrs={
            "Description": "Coarsened land-sea mask",
            "Land": 0,
            "Sea": 1,
            "Original_resolution": "128x256",
            "Coarsened_resolution": f"{H_tgt}x{W_tgt}",
            "min_ocean_fraction": min_ocean_fraction,
        },
    )

    return mask_coarsened
