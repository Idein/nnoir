import chainer.functions as F

class ConvertSoftmax():
    def to_chainer(edge, x):
        return F.softmax(x, edge.params['axis'])
