from chainer.links import BatchNormalization
from chainer.mlir.patch import encode_ndarray, patched_link_call

BatchNormalization.__call__ = patched_link_call(BatchNormalization.__call__)

def to_mlir_node(self):
    gamma = encode_ndarray(self.gamma.data) if (hasattr(self, 'gamma') and self.gamma is not None) else None
    beta = encode_ndarray(self.beta.data) if (hasattr(self, 'beta') and self.beta is not None) else None
    return {
        b'name': 'BatchNormalization',
        b'params': {
            b'eps': self.eps,
            b'avg_mean': encode_ndarray(self.avg_mean),
            b'avg_var': encode_ndarray(self.avg_var),
            b'gamma': gamma,
            b'beta': beta,
        }
    }
BatchNormalization.to_mlir_node = to_mlir_node
