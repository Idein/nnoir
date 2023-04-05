from typing import Any, List, Set

from .function import Function


class Transpose(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"axes"}
        optional_params: Set[str] = set()
        super(Transpose, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):  # type: ignore
        return x.transpose(self.params["axes"])
