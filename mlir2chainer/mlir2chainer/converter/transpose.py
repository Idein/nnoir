import chainer.functions as F

class ConvertTranspose():
    def to_chainer(edge, x):
        return F.transpose(x, edge.params["axes"])
