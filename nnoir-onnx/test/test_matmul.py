import numpy as np
import onnx
import pytest
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_matmul_00():
    """
    opset version >= 9
    """

    shape = (3, 4)
    w_shape = (4, 5)
    out_shape = (3, 5)

    class MatMulTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("MatMul", inputs=["x", "W"], outputs=["y"])
            inputs = [info("x", TensorProto.FLOAT, shape)]
            outputs = [info("y", TensorProto.FLOAT, out_shape)]

            init_W = from_array(np.random.rand(*w_shape).astype(np.float32), "W")

            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[init_W])
            model = make_model(graph)
            return model

    x = np.random.rand(*shape).astype(np.float32)

    outputs = ["y"]
    MatMulTester({"x": x}, outputs).run()


def test_matmul_01():
    """
    opset version >= 9
    """

    shape = (1, 2, 3, 4)
    w_shape = (1, 2, 4, 3)
    out_shape = (1, 2, 3, 3)

    class MatMulTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("MatMul", inputs=["x", "W"], outputs=["y"])
            inputs = [info("x", TensorProto.FLOAT, shape)]
            outputs = [info("y", TensorProto.FLOAT, out_shape)]

            init_W = from_array(np.random.rand(*w_shape).astype(np.float32), "W")

            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[init_W])
            model = make_model(graph)
            return model

    x = np.random.rand(*shape).astype(np.float32)

    outputs = ["y"]
    MatMulTester({"x": x}, outputs).run()


def test_matmul_02():
    """
    opset version >= 9
    """

    in_shape0 = (3, 4)
    in_shape1 = (4, 5)
    out_shape = (3, 5)

    class MatMulTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("MatMul", inputs=["x", "y"], outputs=["z"])
            inputs = [
                info("x", TensorProto.FLOAT, in_shape0),
                info("y", TensorProto.FLOAT, in_shape1),
            ]
            outputs = [info("z", TensorProto.FLOAT, out_shape)]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    x = np.random.rand(*in_shape0).astype(np.float32)
    y = np.random.rand(*in_shape1).astype(np.float32)

    outputs = ["z"]
    MatMulTester({"x": x, "y": y}, outputs).run()


def test_matmul_03():
    """
    opset version >= 9
    """

    in_shape0 = (4,)
    in_shape1 = (4, 5)
    out_shape = (5,)

    class MatMulTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("MatMul", inputs=["x", "y"], outputs=["z"])
            inputs = [
                info("x", TensorProto.FLOAT, in_shape0),
                info("y", TensorProto.FLOAT, in_shape1),
            ]
            outputs = [info("z", TensorProto.FLOAT, out_shape)]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    x = np.random.rand(*in_shape0).astype(np.float32)
    y = np.random.rand(*in_shape1).astype(np.float32)

    outputs = ["z"]
    MatMulTester({"x": x, "y": y}, outputs).run()


def test_matmul_04():
    """
    opset version >= 9
    """

    in_shape0 = (3, 4)
    in_shape1 = (4,)
    out_shape = (3,)

    class MatMulTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("MatMul", inputs=["x", "y"], outputs=["z"])
            inputs = [
                info("x", TensorProto.FLOAT, in_shape0),
                info("y", TensorProto.FLOAT, in_shape1),
            ]
            outputs = [info("z", TensorProto.FLOAT, out_shape)]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    x = np.random.rand(*in_shape0).astype(np.float32)
    y = np.random.rand(*in_shape1).astype(np.float32)

    outputs = ["z"]
    MatMulTester({"x": x, "y": y}, outputs).run()


def test_matmul_05():
    """
    opset version >= 9
    """

    in_shape0 = (1, 2, 3, 4)
    in_shape1 = (1, 2, 4, 5)
    out_shape = (1, 2, 3, 5)

    class MatMulTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("MatMul", inputs=["x", "y"], outputs=["z"])
            inputs = [
                info("x", TensorProto.FLOAT, in_shape0),
                info("y", TensorProto.FLOAT, in_shape1),
            ]
            outputs = [info("z", TensorProto.FLOAT, out_shape)]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    x = np.random.rand(*in_shape0).astype(np.float32)
    y = np.random.rand(*in_shape1).astype(np.float32)

    outputs = ["z"]
    MatMulTester({"x": x, "y": y}, outputs).run()


def test_matmul_06():
    """
    opset version >= 9
    """

    shape = (3, 1, 1)
    w_shape = (1, 1)
    out_shape = (3, 1, 1)

    class MatMulTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("MatMul", inputs=["x", "W"], outputs=["y"])
            inputs = [info("x", TensorProto.FLOAT, shape)]
            outputs = [info("y", TensorProto.FLOAT, out_shape)]

            init_W = from_array(np.random.rand(*w_shape).astype(np.float32), "W")

            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[init_W])
            model = make_model(graph)
            return model

    x = np.random.rand(*shape).astype(np.float32)

    outputs = ["y"]
    MatMulTester({"x": x}, outputs).run()
