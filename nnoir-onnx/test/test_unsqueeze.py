from util import Base
from onnx import TensorProto
from onnx.helper import make_node, make_graph, make_model, make_tensor_value_info, make_tensor, make_opsetid
from onnx.numpy_helper import from_array
import onnx
import numpy as np

info = make_tensor_value_info


def test_unsqueeze_00():
    '''
    opset version >= 1
    '''
    class UnsqueezeTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Unsqueeze", inputs=["x"], outputs=["y"], axes=[0, 3])
            inputs = [info("x", TensorProto.FLOAT, (3, 4))]
            outputs = [info("y", TensorProto.FLOAT, (1, 3, 4, 1))]

            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[])
            return make_model(graph)

    inputs = {"x": (np.random.rand(3, 4).astype(np.float32) * 10.0)}
    outputs = ["y"]
    UnsqueezeTester(inputs, outputs).run()


def test_unsqueeze_01():
    '''
    opset version >= 11
    '''
    class UnsqueezeTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Unsqueeze", inputs=["x"], outputs=["y"], axes=[0, -2])
            inputs = [info("x", TensorProto.FLOAT, (3, 4))]
            outputs = [info("y", TensorProto.FLOAT, (1, 3, 1, 4))]

            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[])
            return make_model(graph)

    inputs = {"x": (np.random.rand(3, 4).astype(np.float32) * 10.0)}
    outputs = ["y"]
    UnsqueezeTester(inputs, outputs).run()
