from typing import Any, Dict, List

import numpy as np
import onnx
import pytest
from numpy.typing import NDArray
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_where_00() -> None:
    class WhereTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Where", inputs=["condition", "x", "y"], outputs=["v2"])
            inputs = [info("x", TensorProto.FLOAT, (2, 4))]
            outputs = [info("v2", TensorProto.FLOAT, (2, 4))]

            init_condition = from_array(
                np.array([[True, True, True, True], [True, True, True, True]]).astype(np.bool_), "condition"
            )
            init_y = from_array(np.random.rand(2, 4).astype(np.float32), "y")

            graph = make_graph([node], "where_graph", inputs, outputs, initializer=[init_condition, init_y])
            return make_model(graph)

    x = np.random.rand(2, 4).astype(np.float32)
    outputs = ["v2"]
    WhereTester({"x": x}, outputs).run()
