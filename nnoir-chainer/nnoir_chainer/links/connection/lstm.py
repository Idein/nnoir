import chainer
from chainer.links import LSTM
from nnoir_chainer.patch import encode_ndarray, patched_link_call
import numpy as np
import nnoir
import nnoir.functions as NNOIR

LSTM.__call__ = patched_link_call(LSTM.__call__)


def to_nnoir_node(self, inputs, outputs):
    lstm_in = inputs[0]
    lstm_out = outputs[0]
    # upward
    upward_inputs = [nnoir.Value(b'v0', np.zeros(lstm_in.shape).astype(lstm_in.dtype))]
    upward_outputs = [nnoir.Value(b'v1', np.zeros((lstm_out.shape[0], 4*lstm_out.shape[1])).astype(lstm_out.dtype))]
    upward_input_names = [x.name for x in upward_inputs]
    upward_output_names = [x.name for x in upward_outputs]
    upward_values = upward_inputs + upward_outputs
    upward_functions_linear_b = encode_ndarray(self.upward.b.data) if (
        hasattr(self, "b") and self.upward.b is not None) else None
    upward_functions = [NNOIR.Linear(upward_input_names,
                                     upward_output_names,
                                     W=encode_ndarray(self.upward.W.data),
                                     b=upward_functions_linear_b)]
    upward = nnoir.NNOIR(b'linear',
                         b'chainer',
                         chainer.__version__,
                         upward_input_names,
                         upward_output_names,
                         upward_values,
                         upward_functions)
    # lateral
    lateral_inputs = [nnoir.Value(b'v0', np.zeros((lstm_out.shape[0],   lstm_out.shape[1])).astype(lstm_out.dtype))]
    lateral_outputs = [nnoir.Value(b'v1', np.zeros((lstm_out.shape[0], 4*lstm_out.shape[1])).astype(lstm_out.dtype))]
    lateral_input_names = [x.name for x in lateral_inputs]
    lateral_output_names = [x.name for x in lateral_outputs]
    lateral_values = lateral_inputs + lateral_outputs
    lateral_functions_linear_b = encode_ndarray(self.lateral.b.data) if (
        hasattr(self, "b") and self.lateral.b is not None) else None
    lateral_functions = [NNOIR.Linear(lateral_input_names,
                                      lateral_output_names,
                                      W=encode_ndarray(self.lateral.W.data),
                                      b=lateral_functions_linear_b)]
    lateral = nnoir.NNOIR(b'linear',
                          b'chainer',
                          chainer.__version__,
                          lateral_input_names,
                          lateral_output_names,
                          lateral_values,
                          lateral_functions)
    # sigmoid
    sigmoid_inputs = [nnoir.Value(b'v0', np.zeros((lstm_out.shape[0], lstm_out.shape[1])).astype(lstm_out.dtype))]
    sigmoid_outputs = [nnoir.Value(b'v1', np.zeros((lstm_out.shape[0], lstm_out.shape[1])).astype(lstm_out.dtype))]
    sigmoid_input_names = [x.name for x in sigmoid_inputs]
    sigmoid_output_names = [x.name for x in sigmoid_outputs]
    sigmoid_values = sigmoid_inputs + sigmoid_outputs
    sigmoid_functions = [NNOIR.Sigmoid(sigmoid_input_names, sigmoid_output_names)]
    sigmoid = nnoir.NNOIR(b'sigmoid',
                          b'chainer',
                          chainer.__version__,
                          sigmoid_input_names,
                          sigmoid_output_names,
                          sigmoid_values,
                          sigmoid_functions)
    # tanh
    tanh_inputs = [nnoir.Value(b'v0', np.zeros((lstm_out.shape[0], lstm_out.shape[1])).astype(lstm_out.dtype))]
    tanh_outputs = [nnoir.Value(b'v1', np.zeros((lstm_out.shape[0], lstm_out.shape[1])).astype(lstm_out.dtype))]
    tanh_input_names = [x.name for x in tanh_inputs]
    tanh_output_names = [x.name for x in tanh_outputs]
    tanh_values = tanh_inputs + tanh_outputs
    tanh_functions = [NNOIR.Tanh(tanh_input_names, tanh_output_names)]
    tanh = nnoir.NNOIR(b'tanh',
                       b'chainer',
                       chainer.__version__,
                       tanh_input_names,
                       tanh_output_names,
                       tanh_values,
                       tanh_functions)
    return NNOIR.LSTM(
        [x.name for x in inputs],
        [x.name for x in outputs],
        upward=upward,
        lateral=lateral,
        activation_input=sigmoid,
        activation_output=sigmoid,
        activation_forget=sigmoid,
        activation_cell=tanh,
        activation_hidden=tanh,
        peephole_input=None,
        peephole_output=None,
        peephole_forget=None,
    )


LSTM.to_nnoir_node = to_nnoir_node
