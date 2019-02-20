from chainer.functions.activation.elu import ELU
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(ELU, 'apply'):
    ELU.apply = patched_function_apply(ELU.apply)
else:
    ELU.__call__ = patched_function_call(ELU.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.ELU(inputs, outputs, alpha=self.alpha)


ELU.to_nnoir_node = to_nnoir_node
