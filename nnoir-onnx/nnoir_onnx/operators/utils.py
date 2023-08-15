import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from nnoir.functions import Function
from numpy.typing import NDArray


class InvalidONNXData(Exception):
    def __init__(self, message: str):
        self.message = message


class UnsupportedONNXOperation(Exception):
    def __init__(self, node: onnx.NodeProto, message: str):
        self.node = node
        self.message = message


class UnknownSizedVariable(Exception):
    def __init__(self, message: str):
        self.message = message


class Op:
    def __init__(self, node: onnx.NodeProto, opset_version: int):
        self.node = node
        self.opset_version = opset_version

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        raise UnsupportedONNXOperation(self.node, "not implemented")


def encode_ndarray(obj: Optional[NDArray[Any]]) -> Optional[Dict[bytes, bytes]]:
    if obj is None:
        return None
    else:
        with io.BytesIO() as out:
            np.save(out, obj.copy())  # type: ignore
            return {b"ndarray": out.getvalue()}


def auto_pad_to_manual_pad(n: int, k: int, s: int, d: int, auto_pad: bytes) -> Tuple[int, int]:
    dk = (k - 1) * d + 1
    if n % s == 0:
        pad = max(dk - s, 0)
    else:
        pad = max(dk - n % s, 0)
    if auto_pad == b"SAME_LOWER":
        pad_before = pad // 2
        pad_after = pad - pad_before
        return (pad_before, pad_after)
    elif auto_pad == b"SAME_UPPER":
        pad_after = pad // 2
        pad_before = pad - pad_after
        return (pad_before, pad_after)
    elif auto_pad == b"VALID":
        return (0, 0)
    else:
        raise "invalid"  # type: ignore


def gen_unregisterd_node_name(env: Dict[str, NDArray[Any]]) -> str:
    for i in range(len(env)):
        candidate = f"v{i}"
        if candidate not in env:
            return candidate

    return f"v{len(env)}"


def register_node(env: Dict[str, NDArray[Any]], name: str, val: NDArray[Any]) -> None:
    env[name] = val
