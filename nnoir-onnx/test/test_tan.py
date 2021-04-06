import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_tan_00():
    class TanTester(Base):
        def __init__(self, inputs, outputs):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Tan", inputs=["v0"], outputs=["v1"])
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 3, 4, 5))]

            graph = make_graph([node], "tan_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = (np.random.rand(1, 3, 4, 5).astype(np.float32) - 0.5) * np.pi * 0.99 + np.random.randint(-10, 10, (1, 3, 4, 5)).astype(
        np.float32
    ) * np.pi

    outputs = ["v1"]
    TanTester({"v0": v0}, outputs).run()
