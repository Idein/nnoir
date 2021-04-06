import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_LSTM_00():
    class LSTMTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            input_size = 2
            batch_size = 3
            hidden_size = 3
            weight_scale = 0.1
            number_of_gates = 4
            seq_length = 1
            num_directions = 1

            node = make_node("LSTM", inputs=["x", "W", "R"], outputs=["y"], hidden_size=hidden_size)

            inputs = [info("x", TensorProto.FLOAT, (seq_length, batch_size, input_size))]
            outputs = [
                info(
                    "y",
                    TensorProto.FLOAT,
                    (seq_length, num_directions, batch_size, hidden_size),
                )
            ]

            W = from_array(
                weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32),
                "W",
            )
            R = from_array(
                weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32),
                "R",
            )

            graph = make_graph([node], "lstm_graph", inputs, outputs, initializer=[W, R])
            print(onnx.helper.printable_graph(graph))
            model = make_model(graph)
            return model

    x = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]).astype(np.float32)
    outputs = ["y"]
    LSTMTester({"x": x}, outputs).run()


def test_LSTM_01():
    class LSTMTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            input_size = 2
            batch_size = 3
            hidden_size = 3
            weight_scale = 0.1
            custom_bias = 0.1
            number_of_gates = 4
            seq_length = 1
            num_directions = 1

            node = make_node(
                "LSTM",
                inputs=["x", "W", "R", "B"],
                outputs=["y", "y_h"],
                hidden_size=hidden_size,
            )

            inputs = [info("x", TensorProto.FLOAT, (seq_length, batch_size, input_size))]
            outputs = [
                info(
                    "y",
                    TensorProto.FLOAT,
                    (seq_length, num_directions, batch_size, hidden_size),
                ),
                info("y_h", TensorProto.FLOAT, (num_directions, batch_size, hidden_size)),
            ]

            W = from_array(
                weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32),
                "W",
            )
            R = from_array(
                weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32),
                "R",
            )
            W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(np.float32)
            R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float32)
            B = from_array(np.concatenate((W_B, R_B), 1), "B")

            graph = make_graph([node], "lstm_graph", inputs, outputs, initializer=[W, R, B])
            print(onnx.helper.printable_graph(graph))
            model = make_model(graph)
            return model

    x = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]).astype(np.float32)
    outputs = ["y", "y_h"]
    LSTMTester({"x": x}, outputs).run()


def test_LSTM_02():
    class LSTMTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            input_size = 2
            batch_size = 3
            hidden_size = 3
            weight_scale = 0.1
            custom_bias = 0.1
            number_of_gates = 4
            seq_length = 1
            num_directions = 1

            node = make_node(
                "LSTM",
                inputs=["x", "W", "R", "B", "sequence_lens", "initial_h", "initial_c"],
                outputs=["y", "y_h", "y_c"],
                hidden_size=hidden_size,
            )

            inputs = [info("x", TensorProto.FLOAT, (seq_length, batch_size, input_size))]
            outputs = [
                info(
                    "y",
                    TensorProto.FLOAT,
                    (seq_length, num_directions, batch_size, hidden_size),
                ),
                info("y_h", TensorProto.FLOAT, (num_directions, batch_size, hidden_size)),
                info("y_c", TensorProto.FLOAT, (num_directions, batch_size, hidden_size)),
            ]

            W = from_array(
                weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32),
                "W",
            )
            R = from_array(
                weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32),
                "R",
            )
            W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(np.float32)
            R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float32)
            B = from_array(np.concatenate((W_B, R_B), 1), "B")

            seq_lens = from_array(np.repeat(seq_length, batch_size).astype(np.int32), "sequence_lens")
            init_h = from_array(np.ones((1, batch_size, hidden_size)).astype(np.float32), "initial_h")
            init_c = from_array(np.ones((1, batch_size, hidden_size)).astype(np.float32), "initial_c")

            graph = make_graph(
                [node],
                "lstm_graph",
                inputs,
                outputs,
                initializer=[W, R, B, seq_lens, init_h, init_c],
            )
            print(onnx.helper.printable_graph(graph))
            model = make_model(graph)
            return model

    x = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]).astype(np.float32)
    outputs = ["y", "y_h", "y_c"]
    LSTMTester({"x": x}, outputs).run()


def test_LSTM_03():
    class LSTMTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            input_size = 2
            batch_size = 3
            hidden_size = 3
            weight_scale = 0.1
            custom_bias = 0.1
            number_of_gates = 4
            number_of_peepholes = 3
            seq_length = 1
            num_directions = 1

            node = make_node(
                "LSTM",
                inputs=[
                    "x",
                    "W",
                    "R",
                    "B",
                    "sequence_lens",
                    "initial_h",
                    "initial_c",
                    "P",
                ],
                outputs=["y", "y_h"],
                hidden_size=hidden_size,
            )

            inputs = [info("x", TensorProto.FLOAT, (seq_length, batch_size, input_size))]
            outputs = [
                info(
                    "y",
                    TensorProto.FLOAT,
                    (seq_length, num_directions, batch_size, hidden_size),
                ),
                info("y_h", TensorProto.FLOAT, (num_directions, batch_size, hidden_size)),
            ]

            W = from_array(
                weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32),
                "W",
            )
            R = from_array(
                weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32),
                "R",
            )
            W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(np.float32)
            R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float32)
            B = from_array(np.concatenate((W_B, R_B), 1), "B")

            seq_lens = from_array(np.repeat(seq_length, batch_size).astype(np.int32), "sequence_lens")
            init_h = from_array(np.ones((1, batch_size, hidden_size)).astype(np.float32), "initial_h")
            init_c = from_array(np.ones((1, batch_size, hidden_size)).astype(np.float32), "initial_c")
            P = from_array(
                weight_scale * np.ones((1, number_of_peepholes * hidden_size)).astype(np.float32),
                "P",
            )

            graph = make_graph(
                [node],
                "lstm_graph",
                inputs,
                outputs,
                initializer=[W, R, B, seq_lens, init_h, init_c, P],
            )
            print(onnx.helper.printable_graph(graph))
            model = make_model(graph)
            return model

    x = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]).astype(np.float32)
    outputs = ["y", "y_h"]
    LSTMTester({"x": x}, outputs).run()


def test_LSTM_04():
    class LSTMTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            input_size = 2
            batch_size = 3
            hidden_size = 3
            weight_scale = 0.1
            number_of_gates = 4
            seq_length = 1
            num_directions = 1

            node = make_node(
                "LSTM",
                inputs=["x", "W", "R"],
                outputs=["y"],
                hidden_size=hidden_size,
                activations=["Sigmoid", "Relu", "Tanh"],
            )

            inputs = [info("x", TensorProto.FLOAT, (seq_length, batch_size, input_size))]
            outputs = [
                info(
                    "y",
                    TensorProto.FLOAT,
                    (seq_length, num_directions, batch_size, hidden_size),
                )
            ]

            W = from_array(
                weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32),
                "W",
            )
            R = from_array(
                weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32),
                "R",
            )

            graph = make_graph([node], "lstm_graph", inputs, outputs, initializer=[W, R])
            print(onnx.helper.printable_graph(graph))
            model = make_model(graph)
            return model

    x = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]).astype(np.float32)
    outputs = ["y"]
    LSTMTester({"x": x}, outputs).run()
