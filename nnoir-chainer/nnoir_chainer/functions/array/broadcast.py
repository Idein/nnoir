from chainer.functions.array.broadcast import BroadcastTo
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(BroadcastTo, 'apply'):
    BroadcastTo.apply = patched_function_apply(BroadcastTo.apply)
else:
    BroadcastTo.__call__ = patched_function_call(BroadcastTo.__call__)


def to_nnoir_node(self, inputs, outputs):
    output = outputs[0]
    return NNOIR.BroadcastTo([x.name for x in inputs], [x.name for x in outputs], shape=outputs.shape)


BroadcastTo.to_nnoir_node = to_nnoir_node
