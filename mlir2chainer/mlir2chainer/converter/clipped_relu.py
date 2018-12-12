import chainer.functions as F

class ConvertClippedReLU():
    def to_chainer(edge, x):
        return F.clipped_relu(x, edge.params['upper'])
