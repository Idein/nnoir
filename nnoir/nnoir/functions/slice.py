from typing import Any, List, Set

import numpy as np

from . import util
from .function import Function


class Slice(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"axes", "starts", "ends"}
        optional_params: Set[str] = set()
        super(Slice, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        axes = self.params["axes"]
        starts = self.params["starts"]
        ends = self.params["ends"]

        indices = [np.s_[:]] * x.ndim
        for i in range(len(axes)):
            indices[axes[i]] = np.s_[starts[i]:ends[i]]
        R = x[tuple(indices)].copy()
        return R


