from typing import Union

import numpy as np
import torch


def linear(
    t: float,
    v0: Union[np.ndarray, torch.Tensor],
    v1: Union[np.ndarray, torch.Tensor],
):
    """Linear interpolation

    Args:
        t (float): Float value between 0.0 and 1.0
        v0 (Union[np.ndarray, torch.Tensor]): A tensor from an adapted model
        v1 (Union[np.ndarray, torch.Tensor]): A tensor from a chat model

    Returns:
        Union[np.ndarray, torch.Tensor]: Interpolated tensor between v0 and v1
    """
    return (1.0 - t) * v0 + t * v1