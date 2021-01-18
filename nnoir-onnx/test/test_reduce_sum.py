import numpy as np

import onnx
from onnx.helper import make_node, make_graph, make_model, make_tensor_value_info, make_opsetid
from onnx import TensorProto
from onnx.numpy_helper import from_array

from util import Base

info = make_tensor_value_info


def test_reduce_sum_00():
    '''
    opset version >= 1
    '''
    class ReduceSumTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("ReduceSum", inputs=["v0"], outputs=["v1"])
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 1, 1, 1))]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph, opset_imports=[make_opsetid("", 11)])
            return model

    v0 = np.random.rand(1, 3, 4, 5).astype(np.float32)

    outputs = ["v1"]
    ReduceSumTester({"v0": v0}, outputs).run()


def test_reduce_sum_01():
    '''
    opset version >= 1
    '''
    class ReduceSumTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("ReduceSum", inputs=["v0"], outputs=["v1"], keepdims=0)
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, [])]  # the shape is scalar

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph, opset_imports=[make_opsetid("", 11)])
            return model

    v0 = np.random.rand(1, 3, 4, 5).astype(np.float32)

    outputs = ["v1"]
    ReduceSumTester({"v0": v0}, outputs).run()


def test_reduce_sum_02():
    '''
    opset version >= 1
    '''
    class ReduceSumTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("ReduceSum", inputs=["v0"], outputs=["v1"], axes=[1, 2])
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, [1, 1, 1, 5])]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph, opset_imports=[make_opsetid("", 11)])
            return model

    v0 = np.random.rand(1, 3, 4, 5).astype(np.float32)

    outputs = ["v1"]
    ReduceSumTester({"v0": v0}, outputs).run()
