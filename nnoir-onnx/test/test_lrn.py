import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_lrn_00():
    """
    opser version >= 1
    """

    class LRNTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "LRN",
                inputs=["v0"],
                outputs=["v1"],
                alpha=0.0002,
                beta=0.6,
                bias=0.8,
                size=3,
            )
            inputs = [info("v0", TensorProto.FLOAT, (5, 5, 5, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (5, 5, 5, 5))]
            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(5, 5, 5, 5).astype(np.float32)

    outputs = ["v1"]
    LRNTester({"v0": v0}, outputs).run()
