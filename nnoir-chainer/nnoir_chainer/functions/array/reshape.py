from chainer.functions.array.reshape import Reshape
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(Reshape, 'apply'):
    Reshape.apply = patched_function_apply(Reshape.apply)
else:
    Reshape.__call__ = patched_function_call(Reshape.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.Reshape([x.name for x in inputs], [x.name for x in outputs], shape=self.shape)


Reshape.to_nnoir_node = to_nnoir_node
