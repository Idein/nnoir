from chainer.functions.pooling.average_pooling_2d import AveragePooling2D
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(AveragePooling2D, 'apply'):
    AveragePooling2D.apply = patched_function_apply(AveragePooling2D.apply)
else:
    AveragePooling2D.__call__ = patched_function_call(AveragePooling2D.__call__)

def to_mlir_node(self):
    return {
        b'name': 'AveragePooling2D',
        b'params': {
            b'kernel': (self.kh, self.kw),
            b'stride': (self.sy, self.sx),
            b'pad_h' : (self.ph, self.ph + self.sy - 1),
            b'pad_w' : (self.pw, self.pw + self.sx - 1),
            b'count_exclude_pad' : False
        }
    }
AveragePooling2D.to_mlir_node = to_mlir_node
