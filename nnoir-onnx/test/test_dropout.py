import numpy as np
import onnx
import pytest
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info

shape = (1, 3, 4, 5)


def test_dropout_00():
    class DropoutTester(Base):
        """
        opset version 10
        """

        def __init__(self, inputs, outputs):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            add_node = make_node("Add", inputs=["v0", "v1"], outputs=["v2"])
            dropout_node = make_node("Dropout", inputs=["v2"], outputs=["v3"], ratio=0.1)
            inputs = [
                info("v0", TensorProto.FLOAT, shape),
                info("v1", TensorProto.FLOAT, shape),
            ]
            outputs = [info("v3", TensorProto.FLOAT, shape)]

            graph = make_graph([add_node, dropout_node], "add_graph", inputs, outputs)
            model = make_model(graph, opset_imports=[make_opsetid("", 10)])
            return model

    v0 = np.random.rand(*shape).astype(np.float32)
    v1 = np.random.rand(*shape).astype(np.float32)

    outputs = ["v3"]
    DropoutTester({"v0": v0, "v1": v1}, outputs).run()


@pytest.mark.xfail()
def test_dropout_01():
    class DropoutTester(Base):
        """
        Consideration: Optional output 'mask'
        """

        def __init__(self, inputs, outputs):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            add_node = make_node("Add", inputs=["v0", "v1"], outputs=["v2"])
            dropout_node = make_node("Dropout", inputs=["v2"], outputs=["v3", "mask"], ratio=0.1)
            inputs = [
                info("v0", TensorProto.FLOAT, shape),
                info("v1", TensorProto.FLOAT, shape),
            ]
            outputs = [
                info("v3", TensorProto.FLOAT, shape),
                info("mask", TensorProto.FLOAT, shape),
            ]

            graph = make_graph([add_node, dropout_node], "add_graph", inputs, outputs)
            model = make_model(graph, opset_imports=[make_opsetid("", 7)])
            return model

    v0 = np.random.rand(*shape).astype(np.float32)
    v1 = np.random.rand(*shape).astype(np.float32)

    outputs = ["v3", "mask"]
    DropoutTester({"v0": v0, "v1": v1}, outputs).run()
