from chainer.functions.activation.relu import ReLU
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(ReLU, 'apply'):
    ReLU.apply = patched_function_apply(ReLU.apply)
else:
    ReLU.__call__ = patched_function_call(ReLU.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.ReLU([x.name for x in inputs], [x.name for x in outputs])


ReLU.to_nnoir_node = to_nnoir_node
