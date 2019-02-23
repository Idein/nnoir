from chainer.links import BatchNormalization
from nnoir_chainer.patch import encode_ndarray, patched_link_call
import nnoir.functions as NNOIR

BatchNormalization.__call__ = patched_link_call(BatchNormalization.__call__)


def to_nnoir_node(self, inputs, outputs):
    gamma = encode_ndarray(self.gamma.data) if (hasattr(self, 'gamma') and self.gamma is not None) else None
    beta = encode_ndarray(self.beta.data) if (hasattr(self, 'beta') and self.beta is not None) else None
    return NNOIR.BatchNormalization(
        [x.name for x in inputs],
        [x.name for x in outputs],
        eps=self.eps,
        avg_mean=encode_ndarray(self.avg_mean),
        avg_var=encode_ndarray(self.avg_var),
        gamma=gamma,
        beta=beta,
    )


BatchNormalization.to_nnoir_node = to_nnoir_node
