from chainer.functions.activation.sigmoid import Sigmoid
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(Sigmoid, 'apply'):
    Sigmoid.apply = patched_function_apply(Sigmoid.apply)
else:
    Sigmoid.__call__ = patched_function_call(Sigmoid.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.Sigmoid([x.name for x in inputs], [x.name for x in outputs])


Sigmoid.to_nnoir_node = to_nnoir_node
