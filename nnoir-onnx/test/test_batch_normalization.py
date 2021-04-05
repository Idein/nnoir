import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info

channel = 3


def test_batch_normalization_00():
    class BatchNormalizationTester(Base):
        def __init__(self, inputs, outputs):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "BatchNormalization",
                inputs=["x", "gamma", "beta", "mean", "var"],
                outputs=["v1"],
            )
            gamma = np.zeros(channel, dtype=np.float32)
            beta = np.zeros(channel, dtype=np.float32)
            mean = np.zeros(channel, dtype=np.float32)
            var = np.zeros(channel, dtype=np.float32)

            gamma[:] = 0.9
            beta[:] = 0.1
            mean[:] = 0.2
            var[:] = 0.8

            node_gamma = make_node(
                "Constant",
                value=make_tensor(name="c0", data_type=TensorProto.FLOAT, dims=(channel,), vals=gamma),
                inputs=[],
                outputs=["gamma"],
            )
            node_beta = make_node(
                "Constant",
                value=make_tensor(name="c1", data_type=TensorProto.FLOAT, dims=(channel,), vals=beta),
                inputs=[],
                outputs=["beta"],
            )
            node_mean = make_node(
                "Constant",
                value=make_tensor(name="c2", data_type=TensorProto.FLOAT, dims=(channel,), vals=mean),
                inputs=[],
                outputs=["mean"],
            )
            node_var = make_node(
                "Constant",
                value=make_tensor(name="c3", data_type=TensorProto.FLOAT, dims=(channel,), vals=var),
                inputs=[],
                outputs=["var"],
            )

            inputs = [info("x", TensorProto.FLOAT, (1, channel, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, channel, 4, 5))]
            graph = make_graph(
                [node_gamma, node_beta, node_mean, node_var, node],
                "add_graph",
                inputs,
                outputs,
            )
            model = make_model(graph)
            return model

    x = np.random.rand(*(1, channel, 4, 5)).astype(np.float32)

    outputs = ["v1"]
    BatchNormalizationTester({"x": x}, outputs).run()


def test_batch_normalization_01():
    class BatchNormalizationTester(Base):
        def __init__(self, inputs, outputs):
            super().__init__(inputs, outputs)

        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "BatchNormalization",
                inputs=["x", "gamma", "beta", "mean", "var"],
                outputs=["v1"],
            )
            gamma = np.zeros(channel, dtype=np.float32)
            beta = np.zeros(channel, dtype=np.float32)
            mean = np.zeros(channel, dtype=np.float32)
            var = np.zeros(channel, dtype=np.float32)

            gamma[:] = 0.9
            beta[:] = 0.1
            mean[:] = 0.2
            var[:] = 0.8

            gamma_init = from_array(gamma, "gamma")
            beta_init = from_array(beta, "beta")
            mean_init = from_array(mean, "mean")
            var_init = from_array(var, "var")

            inputs = [info("x", TensorProto.FLOAT, (1, channel, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, channel, 4, 5))]
            graph = make_graph(
                [node],
                "add_graph",
                inputs,
                outputs,
                initializer=[gamma_init, beta_init, mean_init, var_init],
            )
            model = make_model(graph)
            return model

    x = np.random.rand(*(1, channel, 4, 5)).astype(np.float32)

    outputs = ["v1"]
    BatchNormalizationTester({"x": x}, outputs).run()
