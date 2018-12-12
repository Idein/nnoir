import chainer.functions as F

class ConvertFixedBatchNormalization():
    def to_chainer(self, edge, x, gamma, beta, mean, var):
        return F.fixed_batch_normalization(x, gamma, beta, mean, var)
