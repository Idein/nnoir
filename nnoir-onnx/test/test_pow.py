from typing import Any, Dict, List

import numpy as np
import onnx
import pytest
from numpy.typing import NDArray
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_pow_00() -> None:
    """
    Test for y is constant(=2)
    """

    class PowTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Pow", inputs=["x", "y"], outputs=["z"])
            inputs = [
                info("x", TensorProto.FLOAT, (1, 3, 4, 5)),
            ]
            outputs = [info("z", TensorProto.FLOAT, (1, 3, 4, 5))]

            # constant which dimension is none
            const = from_array(np.array(2.0).astype(np.float32), "y")
            graph = make_graph([node], "pow_graph", inputs, outputs, initializer=[const])
            model = make_model(graph)
            return model

    x = np.random.rand(1, 3, 4, 5).astype(np.float32)
    outputs = ["z"]
    PowTester({"x": x}, outputs).run()


@pytest.mark.xfail()
def test_pow_01() -> None:
    """
    Test for y is constant(!=2)
    Not inplemented yet
    """

    class PowTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Pow", inputs=["x", "y"], outputs=["z"])
            inputs = [
                info("x", TensorProto.FLOAT, (1, 3, 4, 5)),
            ]
            outputs = [info("z", TensorProto.FLOAT, (1, 3, 4, 5))]

            # constant which dimension is none
            const = from_array(np.array(3.0).astype(np.float32), "y")
            graph = make_graph([node], "pow_graph", inputs, outputs, initializer=[const])
            model = make_model(graph)
            return model

    x = np.random.rand(1, 3, 4, 5).astype(np.float32)
    outputs = ["z"]
    PowTester({"x": x}, outputs).run()


@pytest.mark.xfail()
def test_pow_02() -> None:
    """
    Test for y is not constant
    Not inplemented yet
    """
    shape = (3, 4, 5)

    class PowTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Pow", inputs=["x", "y"], outputs=["z"])
            inputs = [
                info("x", TensorProto.FLOAT, shape),
                info("y", TensorProto.FLOAT, shape),
            ]
            outputs = [info("z", TensorProto.FLOAT, shape)]
            graph = make_graph([node], "pow_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(*shape).astype(np.float32)
    v1 = np.random.rand(*shape).astype(np.float32)

    outputs = ["z"]
    PowTester({"x": v0, "y": v1}, outputs).run()
