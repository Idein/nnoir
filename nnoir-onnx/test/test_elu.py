from util import Base
from onnx import TensorProto
from onnx.helper import make_node, make_graph, make_model, make_tensor_value_info, make_tensor, make_opsetid
from onnx.numpy_helper import from_array
import onnx
import numpy as np

info = make_tensor_value_info


def test_elu_00():
    '''
    opser version >= 6
    '''
    class EluTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Elu", inputs=["v0"], outputs=["v1"], alpha=0.3)
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 3, 4, 5))]
            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(1, 3, 4, 5).astype(np.float32)

    outputs = ["v1"]
    EluTester({"v0": v0}, outputs).run()
