from chainer.functions.activation.leaky_relu import LeakyReLU
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(LeakyReLU, 'apply'):
    LeakyReLU.apply = patched_function_apply(LeakyReLU.apply)
else:
    LeakyReLU.__call__ = patched_function_call(LeakyReLU.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.LeakyReLU([x.name for x in inputs], [x.name for x in outputs], slope=self.slope)


LeakyReLU.to_nnoir_node = to_nnoir_node
