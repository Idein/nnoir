from typing import Any, Dict, List

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_pad_00() -> None:
    """
    opset version >= 11
    """

    class PadTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Pad", inputs=["v0", "pads"], outputs=["v1"], mode="constant")
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 5, 6, 7))]

            init_pad = from_array(np.array([0, 1, 1, 1, 0, 1, 1, 1]).astype(np.int64), "pads")

            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[init_pad])
            model = make_model(graph)
            return model

    v0 = np.random.rand(1, 3, 4, 5).astype(np.float32)

    outputs = ["v1"]
    PadTester({"v0": v0}, outputs).run()


def test_pad_01() -> None:
    """
    opset version >= 2
    """

    class PadTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "Pad",
                inputs=["v0"],
                outputs=["v1"],
                mode="constant",
                pads=[0, 1, 1, 1, 0, 1, 1, 1],
                value=0.1,
            )
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 5, 6, 7))]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph, opset_imports=[make_opsetid("", 2)])
            return model

    v0 = np.random.rand(1, 3, 4, 5).astype(np.float32)

    outputs = ["v1"]
    PadTester({"v0": v0}, outputs).run()


def test_pad_02() -> None:
    """
    opset version >= 11 and 0-dimension ndarray value
    """

    class PadTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Pad", inputs=["v0", "pads", "value"], outputs=["v1"], mode="constant")
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 5, 6, 7))]

            init_pad = from_array(np.array([0, 1, 1, 1, 0, 1, 1, 1]).astype(np.int64), "pads")
            pad_value = from_array(np.array(1.0).astype(np.float32), "value")

            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[init_pad, pad_value])
            model = make_model(graph)
            return model

    v0 = np.random.rand(1, 3, 4, 5).astype(np.float32)

    outputs = ["v1"]
    PadTester({"v0": v0}, outputs).run()
