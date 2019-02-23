from chainer.functions.activation.tanh import Tanh
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(Tanh, 'apply'):
    Tanh.apply = patched_function_apply(Tanh.apply)
else:
    Tanh.__call__ = patched_function_call(Tanh.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.Tanh([x.name for x in inputs], [x.name for x in outputs])


Tanh.to_nnoir_node = to_nnoir_node
