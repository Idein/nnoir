from util import Base
from onnx import TensorProto
from onnx.helper import make_node, make_graph, make_model, make_tensor_value_info, make_tensor, make_opsetid
from onnx.numpy_helper import from_array
import onnx
import numpy as np

from nose.tools import raises

info = make_tensor_value_info
+

@raises(Exception)
def test_mul_00():
    '''
    opset version >= 7

    '''

    a_shape = (1, 2, 3, 4)
    b_shape = (1, 2, 3, 4)
    out_shape = (1, 2, 3, 4)

    class MulTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Mul", inputs=["A", "B"], outputs=["C"])
            inputs = [info("A", TensorProto.FLOAT, a_shape)]
            outputs = [info("C", TensorProto.FLOAT, out_shape)]

            init_B = from_array(np.random.rand(*b_shape).astype(np.float32), "B")

            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[init_B])
            model = make_model(graph)
            return model

    a = np.random.rand(*a_shape).astype(np.float32)

    outputs = ["C"]
    MulTester({"A": a}, outputs).run()
