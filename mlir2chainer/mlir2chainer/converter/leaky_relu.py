import chainer.functions as F

class ConvertLeakyReLU():
    def to_chainer(self, edge, x):
        return F.leaky_relu(x, edge.params['slope'])
