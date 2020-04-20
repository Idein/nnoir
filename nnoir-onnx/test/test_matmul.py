from util import Base
from onnx import TensorProto
from onnx.helper import make_node, make_graph, make_model, make_tensor_value_info, make_tensor, make_opsetid
from onnx.numpy_helper import from_array
import onnx
import numpy as np

from nose.tools import raises

info = make_tensor_value_info


def test_matmul_00():
    '''
    opset version >= 9

    Currently, this test will fail due to missing implementations
    '''

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


@raises(Exception)
def test_matmul_01():
    '''
    opset version >= 9

    Currently, this test will fail due to missing implementations
    '''

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
