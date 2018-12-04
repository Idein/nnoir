import chainer.functions as F

class ConvertReshape():
    def to_chainer(edge, x):
        return F.reshape(x, tuple(edge.params["shape"]))

