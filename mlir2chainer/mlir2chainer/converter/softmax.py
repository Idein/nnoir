import chainer.functions as F

class ConvertSoftmax():
    def to_chainer(self, edge, x):
        return F.softmax(x, edge.params['axis'])
