r"""A collection of tools used by different blocks."""

from einops import rearrange
from torch import Tensor
from typing import Optional, Tuple


def reshape(
    hide: str,
    x: Tensor,
    mod: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor], Tuple[int]]:
    r"""Reshape a 5-dimensional tensor for convolutional operations.

    Information:
        Reshapes an input tensor by "hiding" either the spatial dimensions (`H`, `W`)
        or the temporal dimension (`T`) within the batch dimension. This facilitates
        applying 2D or 1D convolutional operations.

    Arguments:
        hide: Dimension to hide in the batch. Options are "space" or "time".
        x: Input tensor of shape (B, C, T, H, W).
        mod: Optional modulation vector of shape (B, D).

    Returns:
        x: Reshaped tensor, where the batch includes the hidden dimensions.
        mod: Reshaped modulation vector, or `None` if not provided.
        original_shape: Original shape of the input tensor (B, C, T, H, W).
    """
    B, C, T, H, W = x.shape

    # Define reshaping rules
    reshape_rules = {
        "space": "b c t h w -> (b h w) c t",
        "time": "b c t h w -> (b t) c h w",
    }
    if hide not in reshape_rules:
        raise ValueError(
            f"ERROR (reshape) - Invalid hide option '{hide}'. Choose 'space' or 'time'."
        )

    # Handle modulation vector if provided
    if mod is not None:
        s = H * W if hide == "space" else T
        mod = rearrange(mod.unsqueeze(1).expand(-1, s, -1), "b s d -> (b s) d")

    return rearrange(x, reshape_rules[hide]), mod, (B, C, T, H, W)


def unshape(
    extract: str,
    x: Tensor,
    shape: Tuple[int],
) -> Tensor:
    r"""Restore the original shape of a 5-dimensional tensor.

    Information
        Reverses a reshaping operation where the spatial or temporal dimensions
        were hidden within the batch dimension. The restored tensor returns to
        its original shape.

    Arguments:
        extract: Dimension to extract from the batch. Options are "space" or "time".
        x: Input tensor.
        shape: Original shape of the tensor before reshaping (B, C, T, H, W).

    Returns:
        Tensor of shape (B, C, T, H, W), restored to its original layout.
    """
    B, C, T, H, W = shape

    # Define unshaping rules
    unshape_rules = {
        "space": "(b h w) c t -> b c t h w",
        "time": "(b t) c h w -> b c t h w",
    }
    if extract not in unshape_rules:
        raise ValueError(
            f"ERROR (unshape) - Invalid hide option '{extract}'. Choose 'space' or 'time'."
        )
    return rearrange(x, unshape_rules[extract], b=B, t=T, h=H, w=W)
