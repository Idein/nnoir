from chainer.functions.activation.softmax import Softmax
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(Softmax, 'apply'):
    Softmax.apply = patched_function_apply(Softmax.apply)
else:
    Softmax.__call__ = patched_function_call(Softmax.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.Softmax([x.name for x in inputs], [x.name for x in outputs], axis=self.axis)


Softmax.to_nnoir_node = to_nnoir_node
