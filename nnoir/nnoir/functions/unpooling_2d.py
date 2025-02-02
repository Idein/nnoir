from typing import Any, List, Set

import numpy as np

from . import util
from .function import Function


class Unpooling2D(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {
            "kh",
            "kw",
            "sy",
            "sx",
            "ph",
            "pw",
            "cover_all",
            "outh",
            "outw",
        }
        optional_params: Set[str] = set()
        super(Unpooling2D, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        col = np.tile(x[:, :, None, None], (1, 1, self.params["kh"], self.params["kw"], 1, 1))  # type: ignore
        R = util.col2im_cpu(
            col,
            (self.params["sy"], self.params["sx"]),
            self.params["ph"],
            self.params["pw"],
            self.params["outh"],
            self.params["outw"],
        )
        return R
