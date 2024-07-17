from typing import Any, Dict, List

import numpy as np
import onnx
import pytest
from numpy.typing import NDArray
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor_value_info, make_tensor
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_gather_00() -> None:
    class GatherTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Gather", inputs=["v0", "indices"], outputs=["v1"], axis=2)
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 5, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 3, 5))]

            init_indices = from_array(np.array(1).astype(np.int64), "indices")

            graph = make_graph([node], "gather_graph", inputs, outputs, initializer=[init_indices])
            return make_model(graph)

    v0 = np.random.rand(1, 3, 5, 5).astype(np.float32)
    outputs = ["v1"]
    GatherTester({"v0": v0}, outputs).run()


def test_gather_01() -> None:
    class GatherTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Gather", inputs=["v0", "indices"], outputs=["v1"], axis=2)
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 5, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 3, 5))]

            init_indices = from_array(np.array(-1).astype(np.int64), "indices")

            graph = make_graph([node], "gather_graph", inputs, outputs, initializer=[init_indices])
            return make_model(graph)

    v0 = np.random.rand(1, 3, 5, 5).astype(np.float32)
    outputs = ["v1"]
    GatherTester({"v0": v0}, outputs).run()

