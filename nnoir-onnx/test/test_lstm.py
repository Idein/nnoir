from util import Base
from onnx import TensorProto
from onnx.helper import make_node, make_graph, make_model, make_tensor_value_info, make_tensor, make_opsetid
from onnx.numpy_helper import from_array
import onnx
import numpy as np

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

            node = make_node(
                'LSTM',
                inputs=['x', 'W', 'R'],
                outputs=['y', 'y_h'],
                hidden_size=hidden_size
            )

            inputs = [info("x", TensorProto.FLOAT, (seq_length,batch_size,input_size))]
            outputs = [info("y", TensorProto.FLOAT, (seq_length,num_directions,batch_size,hidden_size)),
                       info("y_h", TensorProto.FLOAT, (num_directions,batch_size,hidden_size))]

            W = from_array(weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32), "W")
            R = from_array(weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32), "R")

            graph = make_graph([node], "lstm_graph", inputs, outputs, initializer=[W, R])
            print(onnx.helper.printable_graph(graph))
            model = make_model(graph)
            return model

    x = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)
    outputs = ["y", "y_h"]
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
                'LSTM',
                inputs=['x', 'W', 'R', 'B'],
                outputs=['y', 'y_h'],
                hidden_size=hidden_size
            )

            inputs = [info("x", TensorProto.FLOAT, (seq_length,batch_size,input_size))]
            outputs = [info("y", TensorProto.FLOAT, (seq_length,num_directions,batch_size,hidden_size)),
                       info("y_h", TensorProto.FLOAT, (num_directions,batch_size,hidden_size))]

            W = from_array(weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32), "W")
            R = from_array(weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32), "R")
            W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(np.float32)
            R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float32)
            B = from_array(np.concatenate((W_B, R_B), 1), "B")


            graph = make_graph([node], "lstm_graph", inputs, outputs, initializer=[W, R, B])
            print(onnx.helper.printable_graph(graph))
            model = make_model(graph)
            return model

    x = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)
    outputs = ["y", "y_h"]
    LSTMTester({"x": x}, outputs).run()
