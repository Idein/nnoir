from util import Base
from onnx import TensorProto
from onnx.helper import make_node, make_graph, make_model, make_tensor_value_info, make_tensor, make_opsetid
from onnx.numpy_helper import from_array
import onnx
import numpy as np

from nnoir_onnx.operators.utils import UnsupportedONNXOperation
from nose.tools import raises

info = make_tensor_value_info


def test_mul_00():
    '''
    opset version >= 7
    without constant, supports multidirectional broadcasting
    '''

    a_shape = (1, 1, 3, 4)
    b_shape = (1, 2, 3, 1)
    out_shape = (1, 2, 3, 4)

    class MulTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Mul", inputs=["A", "B"], outputs=["C"])
            inputs = [info("A", TensorProto.FLOAT, a_shape), info("B", TensorProto.FLOAT, b_shape)]
            outputs = [info("C", TensorProto.FLOAT, out_shape)]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    a = np.random.rand(*a_shape).astype(np.float32)
    b = np.random.rand(*b_shape).astype(np.float32)

    outputs = ["C"]
    MulTester({"A": a, "B": b}, outputs).run()


@raises(UnsupportedONNXOperation)
def test_mul_01():
    '''
    opset version >= 7
    with one constant, does not support multidirectional broadcasting
    '''

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


def test_mul_02():
    '''
    opset <= 6

    support axis attribute
    '''

    a_shape = (1, 2, 3, 4)
    b_shape = (2, 3)
    out_shape = (1, 2, 3, 4)

    class MulTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Mul", inputs=["A", "B"], outputs=["C"], axis=1)
            inputs = [info("A", TensorProto.FLOAT, a_shape), info("B", TensorProto.FLOAT, b_shape)]
            outputs = [info("C", TensorProto.FLOAT, out_shape)]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph, opset_imports=[make_opsetid("", 6)])
            return model

    a = np.random.rand(*a_shape).astype(np.float32)
    b = np.random.rand(*b_shape).astype(np.float32)

    outputs = ["C"]
    MulTester({"A": a, "B": b}, outputs).run()


def test_gemm_00():
    a_shape = (4, 3)
    b_shape = (5, 4)
    c_shape = (1, 5)

    class GemmTester(Base):
        '''
        opset version >= 11
        '''

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Gemm", inputs=["a", "b", "c"], outputs=["y"],
                             alpha=0.3, beta=0.35, transA=1, transB=1
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

def test_pad_01():
    '''
    opset version >= 2
    '''
    class PadTester(Base):
        def __init__(self, inputs, outputs):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Pad", inputs=["v0"], outputs=["v1"], mode="constant",
                             pads=[0, 1, 1, 1, 0, 1, 1, 1], value=0.1)
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 5, 6, 7))]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph, opset_imports=[make_opsetid("", 2)])
            return model

    v0 = np.random.rand(1, 3, 4, 5).astype(np.float32)

    outputs = ["v1"]
    PadTester({"v0": v0}, outputs).run()
