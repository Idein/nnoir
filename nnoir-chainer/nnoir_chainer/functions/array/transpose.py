from chainer.functions.array.transpose import Transpose
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(Transpose, 'apply'):
    Transpose.apply = patched_function_apply(Transpose.apply)
else:
    Transpose.__call__ = patched_function_call(Transpose.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.Transpose([x.name for x in inputs], [x.name for x in outputs], axes=self.axes)


Transpose.to_nnoir_node = to_nnoir_node
