import chainer.functions as F

class ConvertMulConstant():
    def to_chainer(self, edge, x):
        return x * edge.params['value']
