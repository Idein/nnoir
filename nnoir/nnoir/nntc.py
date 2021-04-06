import io

import msgpack
import numpy as np


class NNTC:
    def __init__(self, model_name, inputs, outputs):
        self.version = 0
        self.model = model_name
        self.test_case = TestCase(inputs, outputs)

    def to_dict(self):
        return {
            b"version": self.version,
            b"model": self.model,
            b"test_case": self.test_case.to_dict(),
        }

    def pack(self):
        return msgpack.packb(self.to_dict(), use_bin_type=False)

    def dump(self, file_name):
        result = self.pack()
        with open(file_name, "w") as f:
            f.buffer.write(result)


class TestCase:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def to_dict(self):
        return {
            b"inputs": [i.to_dict() for i in self.inputs],
            b"outputs": [o.to_dict() for o in self.outputs],
        }


class TestValue:
    def __init__(self, name, ndarray):
        self.name = name
        self.ndarray = ndarray

    def to_dict(self):
        def encode_ndarray(obj):
            x = None
            with io.BytesIO() as out:
                np.save(out, obj.copy())
                x = out.getvalue()
            return x

        return {b"value_name": self.name, b"ndarray": encode_ndarray(self.ndarray)}
