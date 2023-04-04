from typing import Any, List, Set

from . import util
from .function import Function


class Constant(Function):
    def __init__(self, inputs: List[bytes], outputs: List[bytes], **params: Any):
        required_params = {"value"}
        optional_params: Set[str] = set()
        super(Constant, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self):  # type: ignore
        return self.params["value"]
