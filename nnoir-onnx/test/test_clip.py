from util import Base
from onnx import TensorProto
from onnx.helper import make_node, make_graph, make_model, make_tensor_value_info, make_tensor, make_opsetid
from onnx.numpy_helper import from_array
import onnx
import numpy as np

info = make_tensor_value_info


def test_clip_00():
    class ClipTester(Base):
        '''
        IR version == 11
        '''

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Clip", inputs=["x", "min", "max"], outputs=["y"])
            max_init = make_tensor("max", TensorProto.FLOAT, (), np.array([6.0], dtype=np.float32))
            min_init = make_tensor("min", TensorProto.FLOAT, (), np.array([0.0], dtype=np.float32))

            inputs = [info("x", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("y", TensorProto.FLOAT, (1, 3, 4, 5))]
            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[min_init, max_init])
            return make_model(graph)

    inputs = {"x": (np.random.rand(1, 3, 4, 5).astype(np.float32) * 10.0)}
    outputs = ["y"]
    ClipTester(inputs, outputs).run()


def test_clip_01():
    class ClipTester(Base):
        '''
        IR version == 6
        '''

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Clip", inputs=["x"], outputs=["y"], min=0.0, max=6.0)

            inputs = [info("x", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("y", TensorProto.FLOAT, (1, 3, 4, 5))]
            graph = make_graph([node], "add_graph", inputs, outputs)
            return make_model(graph, opset_imports=[make_opsetid("", 6)])

    inputs = {"x": (np.random.rand(1, 3, 4, 5).astype(np.float32) * 10.0)}
    outputs = ["y"]
    ClipTester(inputs, outputs).run()
