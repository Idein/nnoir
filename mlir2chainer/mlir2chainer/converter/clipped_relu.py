import chainer.functions as F

class ConvertClippedReLU():
    def to_chainer(self, edge, x):
        return F.clipped_relu(x, edge.params['upper'])
