from .function import Function
import numpy as np
from . import util


class LSTM(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = set()
        optional_params = set()
        super(LSTM, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        raise "unimplemented"
