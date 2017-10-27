import chainer
import chainer.functions as functions
from chainer.mlir import node
from chainer.mlir.node import encode_ndarray

class SoftmaxCrossEntropy(node.Function, functions.SoftmaxCrossEntropy):
    def __init__(self, *inputs, **dicts):
        super(SoftmaxCrossEntropy, self).__init__(functions.SoftmaxCrossEntropy)
        super(node.Function, self).__init__(*inputs, **dicts)

    def to_mlir_node(self):
        return {
            b'name': self.chainer_node_label,
            b'params': {
                b'normalize': self.normalize,
                b'cache_score': self.cache_score,
            }
        }

def softmax_cross_entropy(x, t, normalize=True, cache_score=True, class_weight=None,
                          ignore_label=-1, reduce='mean', enable_double_backprop=False):
    return SoftmaxCrossEntropy(normalize, cache_score, class_weight, ignore_label, reduce)(x, t)
