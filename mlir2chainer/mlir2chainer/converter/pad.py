import chainer.functions as F

class ConvertConstantPadding():
    def to_chainer(edge, x):
        pad_width = list(map(tuple, edge.params["pads"]))
        return F.pad(x, pad_width, 'constant', constant_values=[edge.params["value"]])
