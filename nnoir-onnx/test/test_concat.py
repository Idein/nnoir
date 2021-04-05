import numpy as np
import onnx
import pytest
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_concat_00():
    class ConcatTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Concat", inputs=["v0", "v1"], outputs=["v2"], axis=1)
            inputs = [
                info("v0", TensorProto.FLOAT, (1, 2, 4, 5)),
                info("v1", TensorProto.FLOAT, (1, 2, 4, 5)),
            ]
            outputs = [info("v2", TensorProto.FLOAT, (1, 4, 4, 5))]
            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(1, 2, 4, 5).astype(np.float32)
    v1 = np.random.rand(1, 2, 4, 5).astype(np.float32)

    outputs = ["v2"]
    ConcatTester({"v0": v0, "v1": v1}, outputs).run()


def test_concat_01():
    class ConcatTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Concat", inputs=["v0", "v1", "v2"], outputs=["v3"], axis=2)
            inputs = [
                info("v0", TensorProto.FLOAT, (1, 2, 9, 5)),
                info("v1", TensorProto.FLOAT, (1, 2, 4, 5)),
                info("v2", TensorProto.FLOAT, (1, 2, 7, 5)),
            ]
            outputs = [info("v3", TensorProto.FLOAT, (1, 2, 20, 5))]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(1, 2, 9, 5).astype(np.float32)
    v1 = np.random.rand(1, 2, 4, 5).astype(np.float32)
    v2 = np.random.rand(1, 2, 7, 5).astype(np.float32)

    outputs = ["v3"]
    ConcatTester({"v0": v0, "v1": v1, "v2": v2}, outputs).run()


@pytest.mark.xfail()
def test_concat_02():
    """
    Test to get value from initializers directly.
    Currently unsupported.
    """

    class ConcatTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Concat", inputs=["v0", "v1", "c"], outputs=["v2"], axis=2)
            inputs = [
                info("v0", TensorProto.FLOAT, (1, 2, 9, 5)),
                info("v1", TensorProto.FLOAT, (1, 2, 4, 5)),
            ]
            outputs = [info("v2", TensorProto.FLOAT, (1, 2, 20, 5))]

            init = from_array(np.random.rand(1, 2, 7, 5).astype(np.float32), "c")

            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[init])
            model = make_model(graph)
            return model

    v0 = np.random.rand(1, 2, 9, 5).astype(np.float32)
    v1 = np.random.rand(1, 2, 4, 5).astype(np.float32)

    outputs = ["v2"]
    ConcatTester({"v0": v0, "v1": v1}, outputs).run()
