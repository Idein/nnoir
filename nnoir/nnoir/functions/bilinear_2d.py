import numpy as np
from .function import Function


class Bilinear2D(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {'size'}
        optional_params = {'mode'}
        super(Bilinear2D, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        out_h = self.params['size'][0]
        out_w = self.params['size'][1]
        batch, ch, in_h, in_w = x.shape
        if 'mode' in self.params and self.params['mode'] == b'align_none':
            # Mode: TF1(default), onnxruntime
            h_scale = float(in_h) / float(out_h)
            w_scale = float(in_w) / float(out_w)
            V = np.minimum(np.arange(out_h) * h_scale, in_h-1).reshape(out_h, 1)
            V0 = np.maximum(0, np.minimum(np.floor(V), in_h - 2)).astype('int32')
            U = np.minimum(np.arange(out_w) * w_scale, in_w-1).reshape(1, out_w)
            U0 = np.maximum(0, np.minimum(np.floor(U), in_w - 2)).astype('int32')
        elif 'mode' not in self.params or self.params['mode'] == b'align_corners':
            # Mode: TF1(align_corners), Chainer, PyTorch
            h_scale = float(in_h-1) / float(out_h-1)
            w_scale = float(in_w-1) / float(out_w-1)
            V = (np.arange(out_h) * h_scale).reshape(out_h, 1)
            V0 = np.maximum(0, np.minimum(np.floor(V), in_h - 2)).astype('int32')
            U = (np.arange(out_w) * w_scale).reshape(1, out_w)
            U0 = np.maximum(0, np.minimum(np.floor(U), in_w - 2)).astype('int32')
        elif 'mode' in self.params and self.params['mode'] == b'align_centers':
            # Mode: TF2, OpenCV
            h_scale = float(in_h) / float(out_h)
            w_scale = float(in_w) / float(out_w)
            V = np.minimum(np.maximum((np.arange(out_h) + 0.5) * h_scale - 0.5, 0.0), float(in_h-1)).reshape(out_h, 1)
            V0 = np.maximum(0, np.minimum(np.floor(V), in_h - 2)).astype('int32')
            U = np.minimum(np.maximum((np.arange(out_w) + 0.5) * w_scale - 0.5, 0.0), float(in_w-1)).reshape(1, out_w)
            U0 = np.maximum(0, np.minimum(np.floor(U), in_w - 2)).astype('int32')
        else:
            raise Exception('unknown Bilinear mode')
        V1 = V0 + 1
        U1 = U0 + 1
        W00 = (U1 - U) * (V1 - V)
        W01 = (U - U0) * (V1 - V)
        W10 = (U1 - U) * (V - V0)
        W11 = (U - U0) * (V - V0)
        V0 = V0.reshape(out_h)
        V1 = np.minimum(V1.reshape(out_h), in_h-1)
        U0 = U0.reshape(out_w)
        U1 = np.minimum(U1.reshape(out_w), in_w-1)
        x0 = x[:, :, V0, :]
        x1 = x[:, :, V1, :]
        result = \
            W00 * x0[:, :, :, U0] + \
            W01 * x0[:, :, :, U1] + \
            W10 * x1[:, :, :, U0] + \
            W11 * x1[:, :, :, U1]
        return result
