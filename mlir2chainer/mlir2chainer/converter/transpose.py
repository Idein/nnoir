import chainer.functions as F

class ConvertTranspose():
    def to_chainer(self, edge, x):
        return F.transpose(x, edge.params['axes'])
