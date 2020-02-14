from chainer.functions.normalization.local_response_normalization import LocalResponseNormalization
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(LocalResponseNormalization, 'apply'):
    LocalResponseNormalization.apply = patched_function_apply(LocalResponseNormalization.apply)
else:
    LocalResponseNormalization.__call__ = patched_function_call(LocalResponseNormalization.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.LocalResponseNormalization(
        [x.name for x in inputs],
        [x.name for x in outputs],
        n=int(self.n),
        k=float(self.k),
        alpha=float(self.alpha),
        beta=float(self.beta)
    )


LocalResponseNormalization.to_nnoir_node = to_nnoir_node
