from typing import Any, Dict, List

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_hard_swish_00() -> None:
    """
    opset version >= 14
    """

    class HardSwishTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("HardSwish", inputs=["v0"], outputs=["v1"])
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 3, 4, 5))]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    outputs = ["v1"]
    v0 = np.random.rand(1, 3, 4, 5).astype(np.float32) * 12.0 - 6.0
    HardSwishTester({"v0": v0}, outputs).run()
