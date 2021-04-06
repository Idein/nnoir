import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def run_opset_11_tester(input_shape, output_shape, axes):
    class UnsqueezeTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            inputs = [info("x", TensorProto.FLOAT, input_shape)]
            outputs = [info("y", TensorProto.FLOAT, output_shape)]

            node = make_node("Unsqueeze", inputs=["x"], outputs=["y"], axes=axes)
            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[])
            return make_model(graph, opset_imports=[make_opsetid("", 11)])

    UnsqueezeTester({"x": (np.random.rand(*input_shape).astype(np.float32) * 10.0)}, ["y"]).run()


def test_opset_1_unsqueeze_00():
    run_opset_11_tester((3, 4), (1, 3, 4, 1), [0, 3])


def test_opset_11_unsqueeze_01():
    run_opset_11_tester((3, 4), (1, 3, 1, 4), [0, -2])


def run_opset_13_tester(input_shape, output_shape, axes):
    class UnsqueezeTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            inputs = [info("x", TensorProto.FLOAT, input_shape)]
            outputs = [info("y", TensorProto.FLOAT, output_shape)]
            initializer = [from_array(np.array(axes, dtype=np.int64), "axes")]

            node = make_node("Unsqueeze", inputs=["x", "axes"], outputs=["y"])
            graph = make_graph([node], "unsqueeze_graph", inputs, outputs, initializer=initializer)
            return make_model(graph, opset_imports=[make_opsetid("", 13)])

    UnsqueezeTester({"x": (np.random.rand(*input_shape).astype(np.float32) * 10.0)}, ["y"]).run()


def test_opset_13_unsqueeze_00():
    run_opset_13_tester((3, 4), (1, 3, 4, 1), [0, 3])


def test_opset_13_unsqueeze_01():
    run_opset_13_tester((3, 4), (1, 3, 1, 4), [0, -2])
