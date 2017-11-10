from chainer.functions import MaxPooling2D
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(MaxPooling2D, 'apply'):
    MaxPooling2D.apply = patched_function_apply(MaxPooling2D.apply)
else:
    MaxPooling2D.__call__ = patched_function_call(MaxPooling2D.__call__)

def to_mlir_node(self):
    return {
        b'name': 'MaxPooling2D',
        b'params': {
            b'kh': self.kh,
            b'kw': self.kw,
            b'sy': self.sy,
            b'sx': self.sx,
            b'ph': self.ph,
            b'pw': self.pw,
            b'cover_all': self.cover_all,
        }
    }
MaxPooling2D.to_mlir_node = to_mlir_node
