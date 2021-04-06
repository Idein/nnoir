import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_Conv_00():
    class ConvTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "Conv",
                inputs=["x", "W", "B"],
                outputs=["y"],
                pads=[1, 1, 1, 1],
                kernel_shape=[3, 3],
            )
            inputs = [info("x", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("y", TensorProto.FLOAT, (1, 7, 4, 5))]

            W = from_array(np.random.rand(7, 3, 3, 3).astype(np.float32), "W")
            B = from_array(np.random.rand(7).astype(np.float32), "B")

            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[W, B])
            model = make_model(graph)
            return model

    x = np.random.rand(1, 3, 4, 5).astype(np.float32)

    outputs = ["y"]
    ConvTester({"x": x}, outputs).run()


def test_Conv_01():
    x_shape = (1, 4, 4, 5)

    class GroupedConvTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "Conv",
                inputs=["x", "W", "B"],
                outputs=["y"],
                pads=[1, 1, 1, 1],
                kernel_shape=[3, 3],
                group=2,
            )
            inputs = [info("x", TensorProto.FLOAT, x_shape)]
            outputs = [info("y", TensorProto.FLOAT, (1, 4, 4, 5))]

            W = from_array(np.random.rand(4, 2, 3, 3).astype(np.float32), "W")
            B = from_array(np.random.rand(4).astype(np.float32), "B")

            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[W, B])
            model = make_model(graph)
            return model

    x = np.random.rand(*x_shape).astype(np.float32)

    outputs = ["y"]
    GroupedConvTester({"x": x}, outputs).run()
