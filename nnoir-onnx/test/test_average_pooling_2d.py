import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from util import Base

info = make_tensor_value_info


def test_average_pooling_2d_00():
    class AveragePoolTester(Base):
        def __init__(self, inputs, outputs):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("AveragePool", inputs=["v0"], outputs=["v1"], kernel_shape=[2, 2])
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 3, 3, 4))]
            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(*(1, 3, 4, 5)).astype(np.float32)

    outputs = ["v1"]
    AveragePoolTester({"v0": v0}, outputs).run()


def test_average_pooling_2d_01():
    class AveragePoolTester(Base):
        def __init__(self, inputs, outputs):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "AveragePool",
                inputs=["v0"],
                outputs=["v1"],
                kernel_shape=[2, 2],
                pads=[0, 0, 1, 1],
            )
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 3, 4, 5))]
            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(*(1, 3, 4, 5)).astype(np.float32)

    outputs = ["v1"]
    AveragePoolTester({"v0": v0}, outputs).run()


def test_average_pooling_2d_02():
    class AveragePoolTester(Base):
        def __init__(self, inputs, outputs):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "AveragePool",
                inputs=["v0"],
                outputs=["v1"],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            )
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 3, 4, 5))]
            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(*(1, 3, 4, 5)).astype(np.float32)

    outputs = ["v1"]
    AveragePoolTester({"v0": v0}, outputs).run()
