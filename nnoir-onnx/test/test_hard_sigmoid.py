import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor_value_info
from util import Base

info = make_tensor_value_info


class HardSigmoidTester(Base):
    def __init__(self, inputs, outputs, alpha: float, beta: float):
        super().__init__(inputs, outputs)
        self.alpha = alpha
        self.beta = beta

    def create_onnx(self) -> onnx.ModelProto:
        node = make_node("HardSigmoid", inputs=["v0"], outputs=["v1"], alpha=self.alpha, beta=self.beta)
        inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
        outputs = [info("v1", TensorProto.FLOAT, (1, 3, 4, 5))]
        graph = make_graph([node], "add_graph", inputs, outputs)
        model = make_model(graph, opset_imports=[make_opsetid("", 6)])
        return model


def test_hard_sigmoid_00():
    alpha = 0.2
    beta = 0.5
    # y = max(0, min(1, 0.2 * x + 0.5))
    #
    # | condition        | result of `alpha * x + beta` | value of y       |
    # | ---              | ---                          | ---              |
    # | x < -2.5         | result < 0                   | 0                |
    # | x > 2.5          | result > 1                   | 1                |
    # | -2.5 < x < 2.5   | 0 <= result <= 1             | alpha * x + beta |
    #
    v0 = np.random.uniform(-7.5, 7.5, (1, 3, 4, 5)).astype(np.float32)
    HardSigmoidTester({"v0": v0}, ["v1"], alpha=alpha, beta=beta).run()


def test_hard_sigmoid_01():
    alpha = 1 / 3
    beta = 3 / 5
    # y = max(0, min(1, (1/3) * x + (3/5)))
    #
    # | condition        | result of `alpha * x + beta` | value of y       |
    # | ---              | ---                          | ---              |
    # | x < -1.8         | result < 0                   | 0                |
    # | x > 1.2          | result > 1                   | 1                |
    # | -1.8 < x < 1.2   | 0 <= result <= 1             | alpha * x + beta |
    #
    v0 = np.random.uniform(-4.8, 4.8, (1, 3, 4, 5)).astype(np.float32)
    HardSigmoidTester({"v0": v0}, ["v1"], alpha=alpha, beta=beta).run()
