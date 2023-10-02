import tempfile
from typing import Any, Dict, List, Optional

import nnoir
import numpy as np
import onnx
import onnxruntime
from nnoir import NNOIR
from nnoir_onnx import ONNX
from numpy.typing import NDArray

epsilon = 0.0001

TMP_REMOVE = True


class Base:
    def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
        self.inputs: Dict[str, NDArray[Any]] = inputs
        self.outputs: List[str] = outputs

        self.onnx: Optional[onnx.ModelProto] = None
        self.nnoir: Optional[NNOIR] = None

    def run(self) -> None:
        self.onnx = self.create_onnx()
        onnx.checker.check_model(self.onnx)
        onnx_result = self.execute_onnx(self.onnx)

        self.nnoir = self.create_nnoir(self.onnx)
        nnoir_result = self.execute_nnoir(self.nnoir)

        for a, b in zip(onnx_result, nnoir_result):
            assert np.all(abs(a - b) < epsilon)

        rerun_result = self.save_and_run(self.nnoir)
        for a, b in zip(rerun_result, nnoir_result):
            assert np.all(abs(a - b) < epsilon)

    def save_and_run(self, model: NNOIR) -> List[NDArray[Any]]:
        with tempfile.NamedTemporaryFile(delete=TMP_REMOVE) as f:
            model.dump(f.name)
            reload_nnoir: NNOIR = nnoir.load(f.name)
            return self.execute_nnoir(reload_nnoir)

    def create_onnx(self) -> onnx.ModelProto:
        # should be override
        assert False

    def execute_onnx(self, model: onnx.ModelProto) -> List[NDArray[Any]]:
        with tempfile.NamedTemporaryFile(delete=TMP_REMOVE) as f:
            onnx.save(model, f.name)
            sess = onnxruntime.InferenceSession(f.name)
            r = sess.run(self.outputs, self.inputs)
            return r  # type: ignore

    def create_nnoir(self, model: onnx.ModelProto) -> NNOIR:
        with tempfile.NamedTemporaryFile(delete=TMP_REMOVE) as f:
            onnx.save(model, f.name)
            return ONNX(f.name).to_NNOIR()

    def execute_nnoir(self, nnoir: NNOIR) -> List[NDArray[Any]]:
        r = nnoir.run(*self.inputs.values())
        return r
