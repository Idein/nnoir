import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def run_opset_11_tester(input_shape, output_shape, axes=None, keepdims=1):
    class ReduceSumTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            kwargs = {
                "inputs": ["v0"],
                "outputs": ["v1"],
                "keepdims": keepdims,
            }
            if axes is not None:
                kwargs["axes"] = axes
            node = make_node("ReduceSum", **kwargs)
            inputs = [info("v0", TensorProto.FLOAT, input_shape)]
            outputs = [info("v1", TensorProto.FLOAT, output_shape)]

            graph = make_graph([node], "add_graph", inputs, outputs)
            return make_model(graph, opset_imports=[make_opsetid("", 11)])

    ReduceSumTester({"v0": np.random.rand(*input_shape).astype(np.float32)}, ["v1"]).run()


def test_opset_11_reduce_sum_00():
    run_opset_11_tester((1, 3, 4, 5), (1, 1, 1, 1))


def test_opset_11_reduce_sum_01():
    run_opset_11_tester((1, 3, 4, 5), (), keepdims=0)


def test_opset_11_reduce_sum_02():
    run_opset_11_tester((1, 3, 4, 5), (1, 1, 1, 5), axes=[1, 2])


def test_opset_11_reduce_sum_03():
    run_opset_11_tester((1, 3, 4, 5), (1, 5), axes=[1, 2], keepdims=0)


def run_opset_13_tester(input_shape, output_shape, axes=None, keepdims=1):
    class ReduceSumTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            kwargs = {
                "inputs": ["v0"] + ["axes" for _ in [None] if axes is not None],
                "outputs": ["v1"],
                "keepdims": keepdims,
            }

            node = make_node("ReduceSum", **kwargs)
            inputs = [info("v0", TensorProto.FLOAT, input_shape)]
            outputs = [info("v1", TensorProto.FLOAT, output_shape)]

            initializer = []
            if axes is not None:
                initializer.append(from_array(np.array(axes, dtype=np.int64), "axes"))

            graph = make_graph([node], "reduce_sum_graph", inputs, outputs, initializer=initializer)
            return make_model(graph, opset_imports=[make_opsetid("", 13)])

    ReduceSumTester({"v0": np.random.rand(*input_shape).astype(np.float32)}, ["v1"]).run()


def test_opset_13_reduce_sum_00():
    run_opset_13_tester((1, 3, 4, 5), (1, 1, 1, 1))


def test_opset_13_reduce_sum_01():
    run_opset_13_tester((1, 3, 4, 5), (), keepdims=0)


def test_opset_13_reduce_sum_02():
    run_opset_13_tester((1, 3, 4, 5), (1, 1, 1, 5), axes=[1, 2])


def test_opset_13_reduce_sum_03():
    run_opset_13_tester((1, 3, 4, 5), (1, 5), axes=[1, 2], keepdims=0)
