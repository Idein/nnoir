from chainer.functions.pooling.max_pooling_2d import MaxPooling2D
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(MaxPooling2D, 'apply'):
    MaxPooling2D.apply = patched_function_apply(MaxPooling2D.apply)
else:
    MaxPooling2D.__call__ = patched_function_call(MaxPooling2D.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.MaxPooling2D(
        [x.name for x in inputs],
        [x.name for x in outputs],
        kernel=(self.kh, self.kw),
        stride=(self.sy, self.sx),
        pad_h=(self.ph, self.ph + self.sy - 1),
        pad_w=(self.pw, self.pw + self.sx - 1)
    )


MaxPooling2D.to_nnoir_node = to_nnoir_node
