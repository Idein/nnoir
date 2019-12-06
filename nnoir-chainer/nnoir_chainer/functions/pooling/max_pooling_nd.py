from chainer.functions.pooling.max_pooling_nd import MaxPoolingND
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(MaxPoolingND, 'apply'):
    MaxPoolingND.apply = patched_function_apply(MaxPoolingND.apply)
else:
    MaxPoolingND.__call__ = patched_function_call(MaxPoolingND.__call__)


def to_nnoir_node(self, inputs, outputs):
    if self.ndim != 2:
        raise Exception('unsupported ndim "{}"'.format(self.ndim))
    return NNOIR.MaxPooling2D(
        [x.name for x in inputs],
        [x.name for x in outputs],
        kernel=tuple(self.ksize[:2]),
        stride=tuple(self.stride[:2]),
        pad_h=(self.pad[0], self.pad[0] + self.stride[0] - 1),
        pad_w=(self.pad[1], self.pad[1] + self.stride[1] - 1),
    )


MaxPoolingND.to_nnoir_node = to_nnoir_node
