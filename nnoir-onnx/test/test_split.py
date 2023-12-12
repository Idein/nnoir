from typing import Any, Dict, List

import numpy as np
import onnx
import pytest
from numpy.typing import NDArray
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_split_trans_axis2() -> None:
    class SplitTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Split", inputs=["v0"], outputs=["v1", "v2"], axis=2)
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 10))]
            outputs = [
                info("v1", TensorProto.FLOAT, (1, 3, 2, 10)),
                info("v2", TensorProto.FLOAT, (1, 3, 2, 10)),
            ]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph, opset_imports=[make_opsetid("", 13)])
            return model

    v0 = np.random.rand(1, 3, 4, 10).astype(np.float32)

    outputs = ["v1", "v2"]
    SplitTester({"v0": v0}, outputs).run()


def test_split_trans_axis3() -> None:
    class SplitTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Split", inputs=["v0"], outputs=["v1", "v2"], axis=3)
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 10))]
            outputs = [
                info("v1", TensorProto.FLOAT, (1, 3, 4, 5)),
                info("v2", TensorProto.FLOAT, (1, 3, 4, 5)),
            ]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph, opset_imports=[make_opsetid("", 13)])
            return model

    v0 = np.random.rand(1, 3, 4, 10).astype(np.float32)

    outputs = ["v1", "v2"]
    SplitTester({"v0": v0}, outputs).run()


def test_split_default_axis() -> None:
    """
    Omit specification of axis.
    If it is ommited, axis is treated as 0.
    """

    class SplitTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Split", inputs=["v0"], outputs=["v1", "v2"])
            inputs = [info("v0", TensorProto.FLOAT, (2, 3, 4))]
            outputs = [
                info("v1", TensorProto.FLOAT, (1, 3, 4)),
                info("v2", TensorProto.FLOAT, (1, 3, 4)),
            ]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph, opset_imports=[make_opsetid("", 13)])
            return model

    v0 = np.random.rand(2, 3, 4).astype(np.float32)

    outputs = ["v1", "v2"]
    SplitTester({"v0": v0}, outputs).run()


def test_split_specify_split() -> None:
    """
    Specify split attribute (opset 11).
    """

    class SplitTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Split", inputs=["v0"], outputs=["v1", "v2", "v3"], axis=3, split=[2, 3, 5])
            inputs = [
                info("v0", TensorProto.FLOAT, (1, 3, 4, 10)),
            ]
            outputs = [
                info("v1", TensorProto.FLOAT, (1, 3, 4, 2)),
                info("v2", TensorProto.FLOAT, (1, 3, 4, 3)),
                info("v3", TensorProto.FLOAT, (1, 3, 4, 5)),
            ]

            graph = make_graph([node], "add_graph", inputs, outputs)
            return make_model(graph, opset_imports=[make_opsetid("", 11)])

    v0 = np.random.rand(1, 3, 4, 10).astype(np.float32)
    outputs = ["v1", "v2", "v3"]
    SplitTester({"v0": v0}, outputs).run()


def test_split_specify_split_13() -> None:
    """
    Specify split input (opset 13).
    """

    class SplitTester(Base):
        def __init__(self, inputs: Dict[str, NDArray[Any]], outputs: List[str]):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("Split", inputs=["v0", "p0"], outputs=["v1", "v2", "v3"], axis=3)
            inputs = [
                info("v0", TensorProto.FLOAT, (1, 3, 4, 10)),
            ]
            node_p0 = make_node(
                "Constant",
                value=make_tensor(
                    name="p0_constant", data_type=TensorProto.INT64, dims=(3,), vals=np.array([2, 3, 5]).astype(np.int64)
                ),
                inputs=[],
                outputs=["p0"],
            )
            outputs = [
                info("v1", TensorProto.FLOAT, (1, 3, 4, 2)),
                info("v2", TensorProto.FLOAT, (1, 3, 4, 3)),
                info("v3", TensorProto.FLOAT, (1, 3, 4, 5)),
            ]

            graph = make_graph([node_p0, node], "add_graph", inputs, outputs)
            return make_model(graph, opset_imports=[make_opsetid("", 13)])

    v0 = np.random.rand(1, 3, 4, 10).astype(np.float32)

    outputs = ["v1", "v2", "v3"]
    SplitTester({"v0": v0}, outputs).run()
