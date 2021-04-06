import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_resize_00():
    """
    opset version >= 11
    """

    class ResizeTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "Resize",
                inputs=["x", "roi", "scales", "sizes"],
                outputs=["y"],
                mode="linear",
                coordinate_transformation_mode="align_corners",
            )
            inputs = [info("x", TensorProto.FLOAT, (1, 3, 10, 10))]
            outputs = [info("y", TensorProto.FLOAT, (1, 3, 5, 5))]

            init_roi = from_array(np.array([]).astype(np.float32), "roi")
            init_scales = from_array(np.array([]).astype(np.float32), "scales")
            init_sizes = from_array(np.array([1, 3, 5, 5]).astype(np.int64), "sizes")

            graph = make_graph(
                [node],
                "add_graph",
                inputs,
                outputs,
                initializer=[init_roi, init_scales, init_sizes],
            )
            return make_model(graph)

    inputs = {"x": (np.random.rand(1, 3, 10, 10).astype(np.float32) * 10.0)}
    outputs = ["y"]
    ResizeTester(inputs, outputs).run()


def test_resize_01():
    """
    opset version >= 11
    """

    class ResizeTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "Resize",
                inputs=["x", "roi", "scales"],
                outputs=["y"],
                mode="linear",
                coordinate_transformation_mode="align_corners",
            )
            inputs = [info("x", TensorProto.FLOAT, (1, 3, 10, 10))]
            outputs = [info("y", TensorProto.FLOAT, (1, 3, 10, 10))]

            init_roi = from_array(np.array([]).astype(np.float32), "roi")
            init_scales = from_array(np.array([1, 1, 1, 1]).astype(np.float32), "scales")

            graph = make_graph(
                [node],
                "add_graph",
                inputs,
                outputs,
                initializer=[init_roi, init_scales],
            )
            return make_model(graph)

    inputs = {"x": (np.random.rand(1, 3, 10, 10).astype(np.float32) * 10.0)}
    outputs = ["y"]
    ResizeTester(inputs, outputs).run()
