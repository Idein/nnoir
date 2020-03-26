import numpy as np

import onnx
from onnx.helper import make_node, make_graph, make_model, make_tensor_value_info
from onnx import TensorProto

from util import Base

info = make_tensor_value_info

shape = (3, 4, 5)


class AddTester(Base):
    def __init__(self, inputs, outputs):
        super().__init__(inputs, outputs)

    def create_onnx(self) -> onnx.ModelProto:
        node = make_node("Add", inputs=["v0", "v1"], outputs=["v2"])
        inputs = [info("v0", TensorProto.FLOAT, shape), info("v1", TensorProto.FLOAT, shape)]
        outputs = [info("v2", TensorProto.FLOAT, shape)]
        graph = make_graph([node], "add_graph", inputs, outputs)
        model = make_model(graph)
        return model


def test_add():
    v0 = np.random.rand(*shape).astype(np.float32)
    v1 = np.random.rand(*shape).astype(np.float32)

    outputs = ["v2"]
    AddTester({"v0": v0, "v1": v1}, outputs).run()
