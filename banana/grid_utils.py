import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

import numpy as np
import matplotlib.pyplot as plt
from random import sample
from typing import Optional, Callable, Union, List, Tuple
import time

from .common import fdtype, fdtype, cdtype, jnp_ndarray, Jnp_ndarray
from .utils import f_cast, i_cast, c_cast
from .logger import Logger
from .baseclasses import BaseUtil


def subsample3DGrid(field: Jnp_ndarray, L_in: int, n_out: int, L_out: int):
    n_in = field.shape[0]
    if n_in == n_out and L_in == L_out:
        return field
    elif (L_in / L_out % 2 == 0) and (n_in / n_out % 2 == 0):
        ratio = L_in // L_out
        beg = n_in // 2 - ratio * n_in // 2
        end = n_in // 2 + ratio * n_in // 2
        step = (end - beg) // n_out
        if step == 0:
            raise ValueError(
                f"Cannot subsamble grid {n_in=} {L_in=} to {n_out=} {L_out=}"
            )

        sl = slice(beg, end, step)
        return field[(sl,) * 3]
    else:
        raise ValueError(f"Cannot subsamble grid {n_in=} {L_in=} to {n_out=} {L_out=}")


def interpolateOn3DGrid(
    field: Jnp_ndarray, positions: Jnp_ndarray, L_grid: fdtype
) -> Jnp_ndarray:
    """

    Parameters
    ----------
    field: Jnp_ndarray :

    positions: Jnp_ndarray :

    L_grid: fdtype :


    Returns
    -------


    """
    return map_coordinates(
        field, field.shape[0] / L_grid * (positions + L_grid / 2), order=1
    )
