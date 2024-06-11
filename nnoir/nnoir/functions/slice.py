from typing import Any, List, Set

import numpy as np

from . import util
from .function import Function


class Slice(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"starts", "ends"}
        optional_params: Set[str] = set()
        super(Slice, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        starts = np.array(self.params["starts"])
        ends = np.array(self.params["ends"])

        indices = [np.s_[starts[i] : ends[i]] for i in range(np.ndim(x))]
        R = x[tuple(indices)].copy()
        return R
