from util import Base
from onnx import TensorProto
from onnx.helper import make_node, make_graph, make_model, make_tensor_value_info, make_tensor, make_opsetid
from onnx.numpy_helper import from_array
import onnx
import numpy as np

info = make_tensor_value_info


def test_relu_00():
    '''
    opset version >= 6
    '''
    class ReluTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Relu", inputs=["x"], outputs=["y"])
            inputs = [info("x", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("y", TensorProto.FLOAT, (1, 3, 4, 5))]

            graph = make_graph([node], "add_graph", inputs, outputs)
            return make_model(graph)

    inputs = {"x": (np.random.rand(1, 3, 4, 5).astype(np.float32) * 10.0)}
    outputs = ["y"]
    ReluTester(inputs, outputs).run()
