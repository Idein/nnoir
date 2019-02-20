import numpy as np
from .function import Function


class Bilinear2D(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'size'}
        optional_params = set()
        super(Bilinear2D, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        out_h = self.params['size'][0]
        out_w = self.params['size'][1]
        batch, ch, in_h, in_w = x.shape
        h_scale = float(in_h-1) / float(out_h-1)
        w_scale = float(in_w-1) / float(out_w-1)
        V = (np.arange(out_h) * h_scale).reshape(out_h, 1)
        V0 = np.maximum(0, np.minimum(np.floor(V), in_h - 2)).astype('int32')
        V1 = V0 + 1
        U = (np.arange(out_w) * w_scale).reshape(1, out_w)
        U0 = np.maximum(0, np.minimum(np.floor(U), in_w - 2)).astype('int32')
        U1 = U0 + 1
        W00 = (U1 - U) * (V1 - V)
        W01 = (U - U0) * (V1 - V)
        W10 = (U1 - U) * (V - V0)
        W11 = (U - U0) * (V - V0)
        x0 = x[:, :, V0.reshape(out_h), :]
        x1 = x[:, :, V1.reshape(out_h), :]
        result = W00 * x0[:, :, :, U0.reshape(out_w)] + \
            W01 * x0[:, :, :, U1.reshape(out_w)] + \
            W10 * x1[:, :, :, U0.reshape(out_w)] + \
            W11 * x1[:, :, :, U1.reshape(out_w)]
        return result
