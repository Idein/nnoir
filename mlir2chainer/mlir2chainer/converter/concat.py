import chainer.functions as F

class ConvertConcat():
    def to_chainer(edge, *xs):
        return F.concat(xs, edge.params['axis'])
