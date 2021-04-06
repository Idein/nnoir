import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def run_opset_11_tester(input_shape, output_shape, axes=None):
    class SqueezeTester(Base):
        def __init__(self, inputs, outputs):
            super(SqueezeTester, self).__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "Squeeze",
                inputs=["x"],
                outputs=["y"],
                **{"axes": axes for _ in [None] if axes is not None},
            )

            inputs = [info("x", TensorProto.FLOAT, input_shape)]
            outputs = [info("y", TensorProto.FLOAT, output_shape)]

            graph = make_graph([node], "squeeze_graph", inputs, outputs)
            return make_model(graph, opset_imports=[make_opsetid("", 11)])

    SqueezeTester({"x": np.ones(input_shape, dtype=np.float32)}, ["y"]).run()


def test_opset_11_squeeze_00():
    run_opset_11_tester((1, 3, 1, 5), (3, 5))


def test_opset_11_squeeze_01():
    run_opset_11_tester((1, 3, 1, 5), (3, 1, 5), axes=[0])


def test_opset_11_squeeze_02():
    run_opset_11_tester((1, 3, 1, 5), (1, 3, 5), axes=[-2])


def test_opset_11_squeeze_03():
    run_opset_11_tester((1, 3, 1, 5), (3, 5), axes=[0, 2])


def test_opset_11_squeeze_04():
    run_opset_11_tester((1, 3, 1, 5), (3, 5), axes=[0, -2])


def test_opset_11_squeeze_05():
    run_opset_11_tester((1, 3, 1, 5), (3, 5), axes=[2, -4])


def run_opset_13_tester(input_shape, output_shape, axes=None):
    class SqueezeTester(Base):
        def __init__(self, inputs, outputs):
            super(SqueezeTester, self).__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "Squeeze",
                inputs=["x"] + ["axes" for _ in [None] if axes is not None],
                outputs=["y"],
            )

            inputs = [info("x", TensorProto.FLOAT, input_shape)]
            outputs = [info("y", TensorProto.FLOAT, output_shape)]

            initializer = []
            if axes is not None:
                initializer.append(from_array(np.array(axes, dtype=np.int64), "axes"))

            graph = make_graph([node], "squeeze_graph", inputs, outputs, initializer=initializer)
            return make_model(graph, opset_imports=[make_opsetid("", 13)])

    SqueezeTester({"x": np.ones(input_shape, dtype=np.float32)}, ["y"]).run()


def test_opset_13_squeeze_00():
    run_opset_13_tester((1, 3, 1, 5), (3, 5))


def test_opset_13_squeeze_01():
    run_opset_13_tester((1, 3, 1, 5), (3, 1, 5), axes=[0])


def test_opset_13_squeeze_02():
    run_opset_13_tester((1, 3, 1, 5), (1, 3, 5), axes=[-2])


def test_opset_13_squeeze_03():
    run_opset_13_tester((1, 3, 1, 5), (3, 5), axes=[0, 2])


def test_opset_13_squeeze_04():
    run_opset_13_tester((1, 3, 1, 5), (3, 5), axes=[0, -2])


def test_opset_13_squeeze_05():
    run_opset_13_tester((1, 3, 1, 5), (3, 5), axes=[2, -4])
