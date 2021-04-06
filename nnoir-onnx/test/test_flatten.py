import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_flatten_00():
    """
    opser version >= 11
    """

    shape = (1, 3, 4, 5)

    class FlattenTester(Base):
        def __init__(self, inputs, outputs, axis):
            super().__init__(inputs, outputs)
            self.axis = axis

            self.new_shape = (1, -1) if axis == 0 else (int(np.prod(shape[0:axis])), -1)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Flatten", inputs=["v0"], outputs=["v1"], axis=self.axis)
            inputs = [info("v0", TensorProto.FLOAT, shape)]
            outputs = [info("v1", TensorProto.FLOAT, self.new_shape)]
            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(*shape).astype(np.float32)

    outputs = ["v1"]

    for axis in range(4):
        FlattenTester({"v0": v0}, outputs, axis).run()
