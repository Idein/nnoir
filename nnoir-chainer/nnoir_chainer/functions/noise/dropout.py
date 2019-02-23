from chainer.functions.noise.dropout import Dropout
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(Dropout, 'apply'):
    Dropout.apply = patched_function_apply(Dropout.apply)
else:
    Dropout.__call__ = patched_function_call(Dropout.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.Dropout([x.name for x in inputs], [x.name for x in outputs])


Dropout.to_nnoir_node = to_nnoir_node
