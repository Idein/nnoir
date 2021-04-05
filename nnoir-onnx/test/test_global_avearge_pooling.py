import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_global_average_pooling_00():
    """
    opset version >= 1
    """

    x_shape = (1, 3, 5, 5)

    class GlobalAveragePoolTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("GlobalAveragePool", inputs=["x"], outputs=["y"])

            inputs = [info("x", TensorProto.FLOAT, x_shape)]
            outputs = [info("y", TensorProto.FLOAT, (1, 3, 1, 1))]

            graph = make_graph([node], "add_graph", inputs, outputs)
            return make_model(graph)

    x = np.random.rand(*x_shape).astype(np.float32)
    inputs = {"x": x}
    outputs = ["y"]
    GlobalAveragePoolTester(inputs, outputs).run()
