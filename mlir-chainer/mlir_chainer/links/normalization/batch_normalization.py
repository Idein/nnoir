from chainer.links import BatchNormalization
from mlir_chainer.patch import encode_ndarray, patched_link_call
import mlir.functions as MLIR

BatchNormalization.__call__ = patched_link_call(BatchNormalization.__call__)


def to_mlir_node(self, inputs, outputs):
    gamma = encode_ndarray(self.gamma.data) if (hasattr(self, 'gamma') and self.gamma is not None) else None
    beta = encode_ndarray(self.beta.data) if (hasattr(self, 'beta') and self.beta is not None) else None
    return MLIR.BatchNormalization(
        inputs,
        outputs,
        eps=self.eps,
        avg_mean=encode_ndarray(self.avg_mean),
        avg_var=encode_ndarray(self.avg_var),
        gamma=gamma,
        beta=beta,
    )


BatchNormalization.to_mlir_node = to_mlir_node
