from chainer.functions.activation.clipped_relu import ClippedReLU
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(ClippedReLU, 'apply'):
    ClippedReLU.apply = patched_function_apply(ClippedReLU.apply)
else:
    ClippedReLU.__call__ = patched_function_call(ClippedReLU.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.ClippedReLU([x.name for x in inputs], [x.name for x in outputs], upper=self.cap)


ClippedReLU.to_nnoir_node = to_nnoir_node
