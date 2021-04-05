import numpy as np
import onnx
import pytest
from nnoir_onnx.operators.utils import UnsupportedONNXOperation
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from util import Base

info = make_tensor_value_info


def test_max_pool_00():
    class MaxPoolTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "MaxPool",
                inputs=["v0"],
                outputs=["v1"],
                kernel_shape=[2, 2],
                strides=[2, 2],
            )
            inputs = [info("v0", TensorProto.FLOAT, (1, 2, 5, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 2, 2, 2))]
            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(1, 2, 5, 5).astype(np.float32)

    outputs = ["v1"]
    MaxPoolTester({"v0": v0}, outputs).run()


@pytest.mark.xfail()
def test_max_pool_01():
    """
    opset version >= 10

    `ceil_mode` != 0 is not supported
    """

    class MaxPoolTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "MaxPool",
                inputs=["v0"],
                outputs=["v1"],
                kernel_shape=[2, 2],
                strides=[2, 2],
                ceil_mode=1,
            )
            inputs = [info("v0", TensorProto.FLOAT, (1, 2, 5, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 2, 3, 3))]
            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(1, 2, 5, 5).astype(np.float32)

    outputs = ["v1"]
    MaxPoolTester({"v0": v0}, outputs).run()


@pytest.mark.xfail(raises=UnsupportedONNXOperation)
def test_max_pool_02():
    """
    opset version >= 11

    `dilations` is not supported
    """

    class MaxPoolTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "MaxPool",
                inputs=["v0"],
                outputs=["v1"],
                kernel_shape=[2, 2],
                strides=[2, 2],
                dilations=[2, 2],
            )
            inputs = [info("v0", TensorProto.FLOAT, (1, 2, 5, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 2, 2, 2))]
            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(1, 2, 5, 5).astype(np.float32)

    outputs = ["v1"]
    MaxPoolTester({"v0": v0}, outputs).run()


def test_max_pool_03():
    """
    opset version >= 10

    `ceil_mode` with defautl value 0 is supported
    """

    class MaxPoolTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "MaxPool",
                inputs=["v0"],
                outputs=["v1"],
                kernel_shape=[2, 2],
                strides=[2, 2],
                ceil_mode=0,
            )
            inputs = [info("v0", TensorProto.FLOAT, (1, 2, 5, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 2, 2, 2))]
            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(1, 2, 5, 5).astype(np.float32)

    outputs = ["v1"]
    MaxPoolTester({"v0": v0}, outputs).run()
