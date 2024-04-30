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


def test_sin_00() -> None:
    class SliceTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Slice", inputs=["v0", "starts", "ends"], outputs=["v1"])
            inputs = [info("v0", TensorProto.FLOAT, (2, 4))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 3))]

            init_starts = from_array(np.array([1, 0]).astype(np.int32), "starts")
            init_ends = from_array(np.array([2, 3]).astype(np.int32), "ends")

            graph = make_graph([node], "slice_graph", inputs, outputs, initializer=[init_starts, init_ends])
            return make_model(graph)

    v0 = np.random.rand(2, 4).astype(np.float32)
    outputs = ["v1"]
    SliceTester({"v0": v0}, outputs).run()


def test_sin_01() -> None:
    class SliceTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Slice", inputs=["v0", "starts", "ends", "axes"], outputs=["v1"])
            inputs = [info("v0", TensorProto.FLOAT, (3, 5, 6))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 5, 6))]

            init_starts = from_array(np.array([1, 0]).astype(np.int32), "starts")
            init_ends = from_array(np.array([-1, 100]).astype(np.int32), "ends")
            init_axes = from_array(np.array([0, 2]).astype(np.int32), "axes")

            graph = make_graph([node], "slice_graph", inputs, outputs, initializer=[init_starts, init_ends, init_axes])
            return make_model(graph)

    v0 = np.random.rand(3, 5, 6).astype(np.float32)
    outputs = ["v1"]
    SliceTester({"v0": v0}, outputs).run()


@pytest.mark.xfail()
def test_sin_02() -> None:
    class SliceTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Slice", inputs=["v0", "starts", "ends", "axes", "steps"], outputs=["v1"])
            inputs = [info("v0", TensorProto.FLOAT, (3, 5, 6))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 5, 6))]

            init_starts = from_array(np.array([1, 0]).astype(np.int32), "starts")
            init_ends = from_array(np.array([-1, 100]).astype(np.int32), "ends")
            init_axes = from_array(np.array([0, 2]).astype(np.int32), "axes")
            init_steps = from_array(np.array([1, 2]).astype(np.int32), "steps")

            graph = make_graph([node], "slice_graph", inputs, outputs, initializer=[init_starts, init_ends, init_axes, init_steps])
            return make_model(graph)

    v0 = np.random.rand(3, 5, 6).astype(np.float32)
    outputs = ["v1"]
    SliceTester({"v0": v0}, outputs).run()
