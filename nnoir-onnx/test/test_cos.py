import numpy as np

import onnx
from onnx.helper import make_node, make_graph, make_model, make_tensor_value_info, make_opsetid
from onnx import TensorProto
from onnx.numpy_helper import from_array

from util import Base

info = make_tensor_value_info


def test_cos_00():

    class CosTester(Base):
        def __init__(self, inputs, outputs):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Cos", inputs=["v0"], outputs=["v1"])
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 3, 4, 5))]

            graph = make_graph([node], "cos_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = (np.random.rand(1, 3, 4, 5).astype(np.float32) - 0.5) * (16 * np.pi)

    outputs = ["v1"]
    CosTester({"v0": v0}, outputs).run()
