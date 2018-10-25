from chainer.functions.pooling.max_pooling_2d import MaxPooling2D
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(MaxPooling2D, 'apply'):
    MaxPooling2D.apply = patched_function_apply(MaxPooling2D.apply)
else:
    MaxPooling2D.__call__ = patched_function_call(MaxPooling2D.__call__)

def to_mlir_node(self):
    return {
        b'name': 'MaxPooling2D',
        b'params': {
            b'kernel': (self.kh, self.kw),
            b'stride': (self.sy, self.sx),
            b'pad_h' : (self.ph, self.ph + self.sy - 1),
            b'pad_w' : (self.pw, self.pw + self.sx - 1)
        }
    }
MaxPooling2D.to_mlir_node = to_mlir_node
