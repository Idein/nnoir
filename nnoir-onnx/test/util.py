from typing import List, Dict, Optional
import tempfile

import onnx
import onnxruntime
import numpy as np

from nnoir import NNOIR
from nnoir_onnx import ONNX

epsilon = 0.0001


class Tester():
    def __init__(self, inputs, outputs):
        self.inputs: Dict[str, np.ndarray] = inputs
        self.outputs: List[str] = outputs

        self.onnx: Optional[onnx.ModelProto] = None
        self.nnoir: Optional[NNOIR] = None

    def run(self):
        self.onnx = self.create_onnx()
        self.nnoir = self.create_nnoir(self.onnx)
        onnx.checker.check_model(self.onnx)

        onnx_result = self.execute_onnx(self.onnx)
        nnoir_result = self.execute_nnoir(self.nnoir)

        for a, b in zip(onnx_result, nnoir_result):
            assert(np.all(abs(a - b)) < epsilon)

    def create_onnx(self) -> onnx.ModelProto:
        # should be override
        assert False

    def execute_onnx(self, model: onnx.ModelProto) -> List[np.ndarray]:
        with tempfile.NamedTemporaryFile() as f:
            onnx.save(model, f.name)
            sess = onnxruntime.InferenceSession(f.name)
            r = sess.run(self.outputs, self.inputs)
            return r

    def create_nnoir(self, model: onnx.ModelProto) -> NNOIR:
        with tempfile.NamedTemporaryFile() as f:
            onnx.save(model, f.name)
            return ONNX(f.name).to_NNOIR()

    def execute_nnoir(self, nnoir) -> List[np.ndarray]:
        r = nnoir.run(*self.inputs.values())
        return r
