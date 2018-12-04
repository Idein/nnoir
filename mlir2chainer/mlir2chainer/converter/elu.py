import chainer.functions as F

class ConvertELU():
    def to_chainer(edge, x):
        return F.elu(x, edge.params["alpha"])
