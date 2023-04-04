from typing import Any, List, Set

import numpy as np
from numpy.typing import NDArray

from . import util
from .function import Function


class AveragePooling2D(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"kernel", "stride", "pad_h", "pad_w", "count_exclude_pad"}
        optional_params: Set[str] = set()
        super(AveragePooling2D, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x: NDArray[Any]) -> NDArray[Any]:
        if self.params["count_exclude_pad"]:
            img, col = util.im2col_cpu(
                x,
                self.params["kernel"],
                self.params["stride"],
                self.params["pad_h"],
                self.params["pad_w"],
            )
            _, mask = util.im2col_cpu(
                np.ones(x.shape, dtype=np.int32),
                self.params["kernel"],
                self.params["stride"],
                self.params["pad_h"],
                self.params["pad_w"],
            )
            return col.sum(axis=(2, 3)) / mask.sum(axis=(2, 3))  # type: ignore
        else:
            img, col = util.im2col_cpu(
                x,
                self.params["kernel"],
                self.params["stride"],
                self.params["pad_h"],
                self.params["pad_w"],
            )
            return col.mean(axis=(2, 3))  # type: ignore
