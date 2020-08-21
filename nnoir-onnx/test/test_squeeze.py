import numpy as np

import onnx
from onnx.helper import make_node, make_graph, make_model, make_tensor_value_info
from onnx.numpy_helper import from_array
from onnx import TensorProto

from util import Base

info = make_tensor_value_info


def test_squeeze_00():
    shape0 = (1, 3, 1, 5)
    shape1 = (3, 5)

    class SqueezeTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                'Squeeze',
                inputs=['x'],
                outputs=['y'],
            )

            inputs = [info("x", TensorProto.FLOAT, shape0)]
            outputs = [info("y", TensorProto.FLOAT, shape1)]

            graph = make_graph([node], "squeeze_graph", inputs, outputs)
            model = make_model(graph)
            return model

    x = np.ones(shape0).astype(np.float32)
    outputs = ["y"]
    SqueezeTester({"x": x}, outputs).run()


def test_squeeze_01():
    shape0 = (1, 3, 1, 5)
    shape1 = (3, 1, 5)

    class SqueezeTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                'Squeeze',
                inputs=['x'],
                outputs=['y'],
                axes=[0],
            )

            inputs = [info("x", TensorProto.FLOAT, shape0)]
            outputs = [info("y", TensorProto.FLOAT, shape1)]

            graph = make_graph([node], "squeeze_graph", inputs, outputs)
            model = make_model(graph)
            return model

    x = np.ones(shape0).astype(np.float32)
    outputs = ["y"]
    SqueezeTester({"x": x}, outputs).run()


def test_squeeze_02():
    shape0 = (1, 3, 1, 5)
    shape1 = (1, 3, 5)

    class SqueezeTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                'Squeeze',
                inputs=['x'],
                outputs=['y'],
                axes=[-2],
            )

            inputs = [info("x", TensorProto.FLOAT, shape0)]
            outputs = [info("y", TensorProto.FLOAT, shape1)]

            graph = make_graph([node], "squeeze_graph", inputs, outputs)
            model = make_model(graph)
            return model

    x = np.ones(shape0).astype(np.float32)
    outputs = ["y"]
    SqueezeTester({"x": x}, outputs).run()
