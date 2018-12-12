import chainer.functions as F

class ConvertLeakyReLU():
    def to_chainer(edge, x):
        return F.leaky_relu(x, edge.params['slope'])
