from .function import Function
import numpy as np
from . import util


class LSTM(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'upward',
                           'lateral',
                           'activation_input',
                           'activation_output',
                           'activation_forget',
                           'activation_cell',
                           'activation_hidden',
                           'peephole_input',
                           'peephole_output',
                           'peephole_forget'}
        optional_params = set()
        super(LSTM, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        raise "unimplemented"
