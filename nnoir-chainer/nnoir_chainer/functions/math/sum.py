from chainer.functions.math.sum import Sum
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(Sum, 'apply'):
    Sum.apply = patched_function_apply(Sum.apply)
else:
    Sum.__call__ = patched_function_call(Sum.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.Sum([x.name for x in inputs], [x.name for x in outputs], axes=self.axis, keepdims=self.keepdims)


Sum.to_nnoir_node = to_nnoir_node
