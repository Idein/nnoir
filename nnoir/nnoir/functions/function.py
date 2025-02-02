import io
from typing import Any, Dict, List, Set, Tuple, Union

import nnoir
import numpy
from numpy.typing import NDArray


class Function(object):
    def __init__(
        self,
        inputs: List[bytes],
        outputs: List[bytes],
        params: Dict[str, Any],
        required_params: Set[str],
        optional_params: Set[str],
    ):
        if required_params - set(params.keys()) != set():
            lacks = ", ".join(required_params - set(params.keys()))
            raise Exception("lack of required parameter: {}".format(lacks))
        if set(params.keys()) - required_params - optional_params != set():
            unknowns = ", ".join(set(params.keys()) - required_params - optional_params)
            raise Exception("unknown parameter: {}".format(unknowns))
        self.inputs = inputs
        self.outputs = outputs
        self.params = params

    def dump(self) -> Dict[bytes, Any]:
        def encode_ndarray(obj: NDArray[Any]) -> Dict[bytes, Any]:
            x = None
            with io.BytesIO() as out:
                numpy.save(out, obj.copy())  # type: ignore
                x = out.getvalue()
            return {b"ndarray": x}

        binary_params = {}
        for k, v in self.params.items():
            if type(v) is numpy.ndarray:
                binary_params[k.encode()] = encode_ndarray(v)
            elif type(v) is nnoir.NNOIR:
                binary_params[k.encode()] = v.to_model()
            else:
                binary_params[k.encode()] = v
        return {
            b"name": self.__class__.__name__,
            b"inputs": self.inputs,
            b"outputs": self.outputs,
            b"params": binary_params,
        }
