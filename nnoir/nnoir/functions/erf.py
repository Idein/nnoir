import math
from typing import Any, List, Set

import numpy as np

from .function import Function


class Erf(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params: Set[str] = set()
        optional_params: Set[str] = set()
        super(Erf, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        return np.vectorize(math.erf)(x).astype(np.float32)
