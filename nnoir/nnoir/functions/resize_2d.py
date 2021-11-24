import numpy as np

from .function import Function


class Resize2D(Function):
    def __init__(self, inputs, outputs, **params):
        required_params = {"size", "interpolation_mode", "coordinate_transformation_mode"}
        optional_params = set()
        super(Resize2D, self).__init__(inputs, outputs, params, required_params, optional_params)

    def run(self, x):
        out_h = self.params["size"][0]
        out_w = self.params["size"][1]
        modes = self.params["interpolation_mode"].decode("utf-8").split("-")
        if modes[0] == "linear":
            return _run_linear(x, out_h, out_w, self.params["coordinate_transformation_mode"])
        elif modes[0] == "nearest":
            return _run_nearest(x, out_h, out_w, self.params["coordinate_transformation_mode"], modes[1])
        else:
            raise Exception("unknow Resize mode")


def _run_linear(x, out_h, out_w, coordinate_transformation_mode):
    batch, ch, in_h, in_w = x.shape
    if coordinate_transformation_mode == b"align_none":
        # Mode: TF1(default), onnxruntime
        h_scale = float(in_h) / float(out_h)
        w_scale = float(in_w) / float(out_w)
        V = np.minimum(np.arange(out_h) * h_scale, in_h - 1).reshape(out_h, 1)
        V0 = np.maximum(0, np.minimum(np.floor(V), in_h - 2)).astype("int32")
        U = np.minimum(np.arange(out_w) * w_scale, in_w - 1).reshape(1, out_w)
        U0 = np.maximum(0, np.minimum(np.floor(U), in_w - 2)).astype("int32")
    elif coordinate_transformation_mode == b"align_corners":
        # Mode: TF1(align_corners), Chainer, PyTorch
        h_scale = float(in_h - 1) / float(out_h - 1)
        w_scale = float(in_w - 1) / float(out_w - 1)
        V = (np.arange(out_h) * h_scale).reshape(out_h, 1)
        V0 = np.maximum(0, np.minimum(np.floor(V), in_h - 2)).astype("int32")
        U = (np.arange(out_w) * w_scale).reshape(1, out_w)
        U0 = np.maximum(0, np.minimum(np.floor(U), in_w - 2)).astype("int32")
    elif coordinate_transformation_mode == b"align_centers":
        # Mode: TF2, OpenCV
        h_scale = float(in_h) / float(out_h)
        w_scale = float(in_w) / float(out_w)
        V = np.minimum(
            np.maximum((np.arange(out_h) + 0.5) * h_scale - 0.5, 0.0),
            float(in_h - 1),
        ).reshape(out_h, 1)
        V0 = np.maximum(0, np.minimum(np.floor(V), in_h - 2)).astype("int32")
        U = np.minimum(
            np.maximum((np.arange(out_w) + 0.5) * w_scale - 0.5, 0.0),
            float(in_w - 1),
        ).reshape(1, out_w)
        U0 = np.maximum(0, np.minimum(np.floor(U), in_w - 2)).astype("int32")
    else:
        raise Exception("unknown Bilinear mode")
    V1 = V0 + 1
    U1 = U0 + 1
    W00 = (U1 - U) * (V1 - V)
    W01 = (U - U0) * (V1 - V)
    W10 = (U1 - U) * (V - V0)
    W11 = (U - U0) * (V - V0)
    V0 = V0.reshape(out_h)
    V1 = np.minimum(V1.reshape(out_h), in_h - 1)
    U0 = U0.reshape(out_w)
    U1 = np.minimum(U1.reshape(out_w), in_w - 1)
    x0 = x[:, :, V0, :]
    x1 = x[:, :, V1, :]
    result = W00 * x0[:, :, :, U0] + W01 * x0[:, :, :, U1] + W10 * x1[:, :, :, U0] + W11 * x1[:, :, :, U1]
    return result


def _round_index(i, nearest_mode):
    if nearest_mode == "floor":
        return int(np.floor(i))
    else:
        raise Exception("unsupported nearest_mode")


def _get_original_index(i, l0, l1, coordinate_transformation_mode, nearest_mode):
    scale = float(l0) / float(l1)
    if coordinate_transformation_mode == b"asymmetric":
        i_ori = i * scale
    else:
        raise Exception("unsupported coordinate_transformation_mode")
    return _round_index(i_ori, nearest_mode)


def _run_nearest(x, out_h, out_w, coordinate_transformation_mode, nearest_mode):
    batch, ch, in_h, in_w = x.shape
    y = np.empty((batch, ch, out_h, out_w), dtype=x.dtype)
    for b in np.arange(batch):
        for c in np.arange(ch):
            for i1 in np.arange(out_h):
                i0 = _get_original_index(i1, in_h, out_h, coordinate_transformation_mode, nearest_mode)
                for j1 in np.arange(out_w):
                    j0 = _get_original_index(j1, in_w, out_w, coordinate_transformation_mode, nearest_mode)
                    y[b, c, i1, j1] = x[b, c, i0, j0]
    return y
