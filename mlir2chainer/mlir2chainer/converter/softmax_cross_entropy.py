import chainer.functions as F

class ConvertLinearFunction():
    def to_chainer(edge, x ,t):
        return F.softmax_cross_entropy(x, t,
                                       normalize = edge.params["normalize"],
                                       cache_score = edge.params["cache_score"])
