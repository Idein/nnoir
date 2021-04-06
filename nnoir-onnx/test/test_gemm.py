import numpy as np
import onnx
import pytest
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_gemm_00():
    a_shape = (4, 3)
    b_shape = (5, 4)
    c_shape = (1, 5)

    class GemmTester(Base):
        """
        opset version >= 11
        """

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "Gemm",
                inputs=["a", "b", "c"],
                outputs=["y"],
                alpha=0.3,
                beta=0.35,
                transA=1,
                transB=1,
            )

            inputs = [info("a", TensorProto.FLOAT, a_shape)]
            outputs = [info("y", TensorProto.FLOAT, (3, 5))]

            b = np.random.ranf(b_shape).astype(np.float32)
            c = np.random.ranf(c_shape).astype(np.float32)

            b_init = from_array(b, "b")
            c_init = from_array(c, "c")
            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[b_init, c_init])
            return make_model(graph)

    a = np.random.ranf([4, 3]).astype(np.float32)

    inputs = {"a": a}
    outputs = ["y"]
    GemmTester(inputs, outputs).run()


@pytest.mark.xfail()
def test_gemm_01():
    """
    unidirectional broadcasting is not supported
    """
    a_shape = (4, 3)
    b_shape = (5, 4)
    c_shape = (3, 5)

    class GemmTester(Base):
        """
        opset version >= 11
        """

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "Gemm",
                inputs=["a", "b", "c"],
                outputs=["y"],
                alpha=0.3,
                beta=0.35,
                transA=1,
                transB=1,
            )

            inputs = [info("a", TensorProto.FLOAT, a_shape)]
            outputs = [info("y", TensorProto.FLOAT, (3, 5))]

            b = np.random.ranf(b_shape).astype(np.float32)
            c = np.random.ranf(c_shape).astype(np.float32)

            b_init = from_array(b, "b")
            c_init = from_array(c, "c")
            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[b_init, c_init])
            return make_model(graph)

    a = np.random.ranf([4, 3]).astype(np.float32)

    inputs = {"a": a}
    outputs = ["y"]
    GemmTester(inputs, outputs).run()


def test_gemm_02():
    a_shape = (4, 3)
    b_shape = (5, 4)
    c_shape = (3, 5)

    class GemmTester(Base):
        """
        opset version >= 11
        """

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "Gemm",
                inputs=["a", "b", "c"],
                outputs=["y"],
                alpha=0.3,
                beta=0.35,
                transA=1,
                transB=1,
            )

            inputs = [
                info("a", TensorProto.FLOAT, a_shape),
                info("b", TensorProto.FLOAT, b_shape),
                info("c", TensorProto.FLOAT, c_shape),
            ]
            outputs = [info("y", TensorProto.FLOAT, (3, 5))]

            graph = make_graph([node], "gemm_graph", inputs, outputs)
            return make_model(graph)

    a = np.random.ranf(list(a_shape)).astype(np.float32)
    b = np.random.ranf(list(b_shape)).astype(np.float32)
    c = np.random.ranf(list(c_shape)).astype(np.float32)

    inputs = {"a": a, "b": b, "c": c}
    outputs = ["y"]
    GemmTester(inputs, outputs).run()


def test_gemm_03():
    a_shape = (3, 4)
    b_shape = (4, 5)
    c_shape = (3, 5)

    class GemmTester(Base):
        """
        opset version >= 11
        """

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "Gemm",
                inputs=["a", "b", "c"],
                outputs=["y"],
                alpha=0.3,
                beta=0.35,
                transA=0,
                transB=0,
            )

            inputs = [
                info("a", TensorProto.FLOAT, a_shape),
                info("b", TensorProto.FLOAT, b_shape),
                info("c", TensorProto.FLOAT, c_shape),
            ]
            outputs = [info("y", TensorProto.FLOAT, (3, 5))]

            graph = make_graph([node], "gemm_graph", inputs, outputs)
            return make_model(graph)

    a = np.random.ranf(list(a_shape)).astype(np.float32)
    b = np.random.ranf(list(b_shape)).astype(np.float32)
    c = np.random.ranf(list(c_shape)).astype(np.float32)

    inputs = {"a": a, "b": b, "c": c}
    outputs = ["y"]
    GemmTester(inputs, outputs).run()


def test_gemm_04():
    a_shape = (4, 3)
    b_shape = (4, 5)

    class GemmTester(Base):
        """
        opset version >= 11
        """

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "Gemm",
                inputs=["a", "b"],
                outputs=["y"],
                alpha=0.3,
                beta=0.35,
                transA=1,
                transB=0,
            )

            inputs = [
                info("a", TensorProto.FLOAT, a_shape),
                info("b", TensorProto.FLOAT, b_shape),
            ]
            outputs = [info("y", TensorProto.FLOAT, (3, 5))]

            graph = make_graph([node], "gemm_graph", inputs, outputs)
            return make_model(graph)

    a = np.random.ranf(list(a_shape)).astype(np.float32)
    b = np.random.ranf(list(b_shape)).astype(np.float32)

    inputs = {"a": a, "b": b}
    outputs = ["y"]
    GemmTester(inputs, outputs).run()
