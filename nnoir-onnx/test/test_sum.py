import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from util import Base

info = make_tensor_value_info


def test_sum_00():
    shape = (3, 4, 5)

    class SumTester(Base):
        def __init__(self, inputs, outputs):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Sum", inputs=["v0", "v1"], outputs=["v2"])
            inputs = [
                info("v0", TensorProto.FLOAT, shape),
                info("v1", TensorProto.FLOAT, shape),
            ]
            outputs = [info("v2", TensorProto.FLOAT, shape)]
            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(*shape).astype(np.float32)
    v1 = np.random.rand(*shape).astype(np.float32)

    outputs = ["v2"]
    SumTester({"v0": v0, "v1": v1}, outputs).run()


def test_sum_01():
    """
    Test for multidirectional broadcasting
    https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
    """

    class SumTester(Base):
        def __init__(self, inputs, outputs):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Sum", inputs=["v0", "v1"], outputs=["v2"])
            inputs = [
                info("v0", TensorProto.FLOAT, (1, 3, 4, 5)),
                info("v1", TensorProto.FLOAT, []),
            ]
            outputs = [info("v2", TensorProto.FLOAT, (1, 3, 4, 5))]
            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(1, 3, 4, 5).astype(np.float32)
    v1 = np.random.rand(1).astype(np.float32)

    outputs = ["v2"]
    SumTester({"v0": v0, "v1": v1}, outputs).run()


def test_sum_02():
    """
    Test for multidirectional broadcasting
    https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
    """

    class SumTester(Base):
        def __init__(self, inputs, outputs):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Sum", inputs=["v0", "v1"], outputs=["v2"])
            inputs = [
                info("v0", TensorProto.FLOAT, (1, 3, 4, 5)),
                info("v1", TensorProto.FLOAT, [1, 1, 4, 5]),
            ]
            outputs = [info("v2", TensorProto.FLOAT, (1, 3, 4, 5))]
            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(1, 3, 4, 5).astype(np.float32)
    v1 = np.random.rand(1, 1, 4, 5).astype(np.float32)

    outputs = ["v2"]
    SumTester({"v0": v0, "v1": v1}, outputs).run()
