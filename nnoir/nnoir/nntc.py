import io
from typing import Any, Dict, List

import msgpack
import numpy as np
from numpy.typing import NDArray


class TestValue:
    def __init__(self, name: bytes, ndarray: NDArray[Any]):
        self.name = name
        self.ndarray = ndarray

    def to_dict(self) -> Dict[bytes, Any]:
        def encode_ndarray(obj: NDArray[Any]) -> bytes:
            x = None
            with io.BytesIO() as out:
                np.save(out, obj.copy())  # type: ignore
                x = out.getvalue()
            return x

        return {b"value_name": self.name, b"ndarray": encode_ndarray(self.ndarray)}


class TestCase:
    def __init__(self, inputs: List[TestValue], outputs: List[TestValue]):
        self.inputs = inputs
        self.outputs = outputs

    def to_dict(self) -> Dict[bytes, Any]:
        return {
            b"inputs": [i.to_dict() for i in self.inputs],
            b"outputs": [o.to_dict() for o in self.outputs],
        }


class NNTC:
    def __init__(self, model_name: bytes, inputs: List[TestValue], outputs: List[TestValue]):
        self.version = 0
        self.model = model_name
        self.test_case = TestCase(inputs, outputs)

    def to_dict(self) -> Dict[bytes, Any]:
        return {
            b"version": self.version,
            b"model": self.model,
            b"test_case": self.test_case.to_dict(),
        }

    def pack(self) -> bytes:
        return msgpack.packb(self.to_dict(), use_bin_type=False)  # type: ignore

    def dump(self, file_name: str) -> None:
        result = self.pack()
        with open(file_name, "w") as f:
            f.buffer.write(result)


def _load_test_value(v: Dict[bytes, Any]) -> TestValue:
    name = v[b"value_name"]
    arr = np.load(io.BytesIO(v[b"ndarray"]))  # type: ignore
    return TestValue(name, arr)


def load_nntc(file_name: str) -> NNTC:
    with open(file_name, "rb") as f:
        nntc = msgpack.unpackb(f.read(), raw=True)
    version = nntc[b"version"]
    model = nntc[b"model"]
    inputs = [_load_test_value(v) for v in nntc[b"test_case"][b"inputs"]]
    outputs = [_load_test_value(v) for v in nntc[b"test_case"][b"outputs"]]
    assert version == 0
    return NNTC(model, inputs, outputs)
