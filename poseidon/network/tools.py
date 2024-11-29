r"""A collection of tools used by different architecture blocks."""

from einops import rearrange
from torch import Tensor
from typing import Optional, Tuple


def reshape(
    convolution: str,
    x: Tensor,
    mod: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tuple[int]]:
    """Reshape a 5D input before a 1D/2D convolution.

    Information:
        Temporal: B, C, T, X, Y -> (B * X * Y), C, T
        Spatial:  B, C, T, X, Y -> (B * T), C, X, Y

    Arguments:
        convolution: Type of reshaping.
        x: Input tensor, with shape (B, C, T, H, W).
        mod: Modulation vector, with shape (B, D).
    """
    B, C, T, H, W = x.shape

    if convolution == "spatial":
        x = rearrange(x, "b c t h w -> (b t) c h w")
    elif convolution == "temporal":
        x = rearrange(x, "b c t h w -> (b h w) c t")
    else:
        raise NotImplementedError()

    if mod is not None:
        s = H * W if convolution == "spatial" else T
        mod = mod.unsqueeze(1).expand(-1, s, -1)
        mod = rearrange(mod, "b s d -> (b s) d")

    return x, mod, (B, C, T, H, W)


def unshape(
    convolution: str,
    x: Tensor,
    shape: Tuple[int],
) -> Tensor:
    """Unshape a 5D input after a 1D/2D convolution.

    Information:
        Temporal: (B * X * Y), C, T -> B, C, T, X, Y
        Spatial:  (B * T), C, X, Y -> B, C, T, X, Y

    Arguments:
        convolution: Type of reshaping.
        x: Input tensor, with shape (B, C, T, H, W).
        shape: Shape of the input tensor before reshaping.
    """
    B, C, T, H, W = shape

    if convolution == "spatial":
        x = rearrange(x, "(b t) c h w -> b c t h w", b=B, t=T)
    elif convolution == "temporal":
        x = rearrange(x, "(b h w) c t -> b c t h w", b=B, h=H, w=W)
    else:
        raise NotImplementedError()

    return x
