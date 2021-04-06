import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_softmax_00():
    class SoftmaxTester(Base):
        def __init__(self, inputs, outputs):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Softmax", inputs=["v0"], outputs=["v1"], axis=1)
            inputs = [info("v0", TensorProto.FLOAT, (2, 60))]
            outputs = [info("v1", TensorProto.FLOAT, (2, 60))]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(2, 60).astype(np.float32)

    outputs = ["v1"]
    SoftmaxTester({"v0": v0}, outputs).run()


def test_softmax_01():
    class SoftmaxTester(Base):
        def __init__(self, inputs, outputs):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Softmax", inputs=["v0"], outputs=["v1"], axis=0)
            inputs = [info("v0", TensorProto.FLOAT, (60,))]
            outputs = [info("v1", TensorProto.FLOAT, (60,))]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(60).astype(np.float32)

    outputs = ["v1"]
    SoftmaxTester({"v0": v0}, outputs).run()
