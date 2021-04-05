import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_prelu_00():
    class PReluTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("PRelu", inputs=["x", "slope"], outputs=["y"])

            inputs = [info("x", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("y", TensorProto.FLOAT, (1, 3, 4, 5))]

            slope_init = from_array(np.random.rand(1).astype(np.float32), "slope")
            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[slope_init])
            return make_model(graph)

    inputs = {"x": (np.random.rand(1, 3, 4, 5).astype(np.float32) * 10.0)}
    outputs = ["y"]
    PReluTester(inputs, outputs).run()
