from chainer.functions.pooling.average_pooling_2d import AveragePooling2D
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.functions as MLIR

if hasattr(AveragePooling2D, 'apply'):
    AveragePooling2D.apply = patched_function_apply(AveragePooling2D.apply)
else:
    AveragePooling2D.__call__ = patched_function_call(AveragePooling2D.__call__)


def to_mlir_node(self, inputs, outputs):
    return MLIR.AveragePooling2D(
        inputs,
        outputs,
        kernel=(self.kh, self.kw),
        stride=(self.sy, self.sx),
        pad_h=(self.ph, self.ph + self.sy - 1),
        pad_w=(self.pw, self.pw + self.sx - 1),
        count_exclude_pad=False
    )


AveragePooling2D.to_mlir_node = to_mlir_node
