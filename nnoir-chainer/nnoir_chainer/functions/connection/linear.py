from chainer.functions.connection.linear import LinearFunction
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(LinearFunction, 'apply'):
    LinearFunction.apply = patched_function_apply(LinearFunction.apply)
else:
    LinearFunction.__call__ = patched_function_call(LinearFunction.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.LinearFunction(inputs, outputs)


LinearFunction.to_nnoir_node = to_nnoir_node
