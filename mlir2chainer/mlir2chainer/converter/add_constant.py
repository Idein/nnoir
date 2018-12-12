import chainer.functions as F

class ConvertAddConstant():
    def to_chainer(self, edge, x):
        return x + edge.params['value']
