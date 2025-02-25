#!/usr/bin/env python3

import os
from typing import NewType, TypeVar, Union, Dict, Optional
import jax.numpy as jnp
import jax
import numpy as np

# dtype = 'float64'
# cdtype = 'complex128'
fdtype = jnp.float32
cdtype = jnp.complex64
idtype = jnp.int32
# fdtype = jnp.float64
# cdtype = jnp.complex128
# idtype = jnp.int64

jnp_ndarray = type(jnp.array([]))
Jnp_ndarray = jax.typing.ArrayLike  # NewType("Jnp_ndarray", jnp_ndarray)

TypingType = TypeVar("TypingType")
DataType = np.ndarray
ScalarType = Union[float, int]
DictScalarType = Dict[str, ScalarType]
OptionalDataType = Optional[DataType]
OptionalScalarType = Optional[ScalarType]

try:
    banana_path = os.environ["BANANA_PATH"]
except KeyError:
    raise KeyError(
        "BANANA_PATH is not an environment variable, "
        "Please define where you've put the code. "
        "Add to your .bashrc (or equivalent) something like: "
        "set BANANA_PATH /home/name/code/banana/banana/"
    )
