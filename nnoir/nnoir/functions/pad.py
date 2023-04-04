from typing import Any, List, Set

import numpy as np

from .function import Function


class ConstantPadding(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"pads", "value"}
        optional_params: Set[str] = set()
        super(ConstantPadding, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        return np.pad(  # type: ignore
            x,
            self.params["pads"],
            mode="constant",
            constant_values=(self.params["value"],),
        )
