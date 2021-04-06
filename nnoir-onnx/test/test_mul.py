import numpy as np
import onnx
from nnoir_onnx.operators.utils import UnsupportedONNXOperation
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_mul_00():
    """
    opset version >= 7
    without constant, supports multidirectional broadcasting
    """

    a_shape = (1, 1, 3, 4)
    b_shape = (1, 2, 3, 1)
    out_shape = (1, 2, 3, 4)

    class MulTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Mul", inputs=["A", "B"], outputs=["C"])
            inputs = [
                info("A", TensorProto.FLOAT, a_shape),
                info("B", TensorProto.FLOAT, b_shape),
            ]
            outputs = [info("C", TensorProto.FLOAT, out_shape)]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    a = np.random.rand(*a_shape).astype(np.float32)
    b = np.random.rand(*b_shape).astype(np.float32)

    outputs = ["C"]
    MulTester({"A": a, "B": b}, outputs).run()


def test_mul_01():
    """
    opset version >= 7
    with one constant, unidirectional broadcasting (from constant to variable)
    """

    a_shape = (1, 2, 3, 4)
    b_shape = (1, 1, 3, 1)
    out_shape = (1, 2, 3, 4)

    class MulTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Mul", inputs=["A", "B"], outputs=["C"])
            inputs = [info("A", TensorProto.FLOAT, a_shape)]
            outputs = [info("C", TensorProto.FLOAT, out_shape)]

            B = np.random.rand(*b_shape).astype(np.float32)

            b_init = from_array(B, "B")
            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[b_init])
            model = make_model(graph)
            return model

    a = np.random.rand(*a_shape).astype(np.float32)

    outputs = ["C"]
    MulTester({"A": a}, outputs).run()


def test_mul_02():
    """
    opset version >= 7
    with one constant, support multidirectional broadcasting
    """

    a_shape = (1, 1, 3, 4)
    b_shape = (1, 2, 3, 1)
    out_shape = (1, 2, 3, 4)

    class MulTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Mul", inputs=["A", "B"], outputs=["C"])
            inputs = [info("A", TensorProto.FLOAT, a_shape)]
            outputs = [info("C", TensorProto.FLOAT, out_shape)]

            B = np.random.rand(*b_shape).astype(np.float32)

            b_init = from_array(B, "B")
            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[b_init])
            model = make_model(graph)
            return model

    a = np.random.rand(*a_shape).astype(np.float32)

    outputs = ["C"]
    MulTester({"A": a}, outputs).run()


def test_mul_03():
    """
    opset version >= 7
    with one constant, different shape length
    """

    a_shape = (1, 2, 3, 4)
    b_shape = (3, 4)
    out_shape = (1, 2, 3, 4)

    class MulTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Mul", inputs=["A", "B"], outputs=["C"])
            inputs = [info("A", TensorProto.FLOAT, a_shape)]
            outputs = [info("C", TensorProto.FLOAT, out_shape)]

            B = np.random.rand(*b_shape).astype(np.float32)

            b_init = from_array(B, "B")
            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[b_init])
            model = make_model(graph)
            return model

    a = np.random.rand(*a_shape).astype(np.float32)

    outputs = ["C"]
    MulTester({"A": a}, outputs).run()
