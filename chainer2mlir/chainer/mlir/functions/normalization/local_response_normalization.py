import chainer
import chainer.functions as functions
from chainer.mlir import node
from chainer.mlir.node import encode_ndarray

class LocalResponseNormalization(node.Function, functions.LocalResponseNormalization):
    def __init__(self, *inputs, **dicts):
        super(LocalResponseNormalization, self).__init__(functions.LocalResponseNormalization)
        super(node.Function, self).__init__(*inputs, **dicts)

    def to_mlir_node(self):
        return {
            b'name': self.chainer_node_label,
            b'params': {
                b'n': self.b,
                b'k': self.k,
                b'alpha': self.alpha,
                b'beta': self.beta
            }
        }

def local_response_normalization(x, n=5, k=2, alpha=1e-4, beta=.75):
    return LocalResponseNormalization(n, k, alpha, beta)(x)
