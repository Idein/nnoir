from chainer.functions.array.concat import Concat
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(Concat, 'apply'):
    Concat.apply = patched_function_apply(Concat.apply)
else:
    Concat.__call__ = patched_function_call(Concat.__call__)


def to_nnoir_node(self, inputs, outputs):
    self.axis = len(outputs[0].shape) + self.axis if self.axis < 0 else self.axis
    return NNOIR.Concat([x.name for x in inputs], [x.name for x in outputs], axis=int(self.axis))


Concat.to_nnoir_node = to_nnoir_node
