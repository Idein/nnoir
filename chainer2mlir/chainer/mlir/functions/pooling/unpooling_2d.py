from chainer.functions.pooling.unpooling_2d import Unpooling2D
from chainer.mlir.patch import patched_function_apply, patched_function_call

if hasattr(Unpooling2D, 'apply'):
    Unpooling2D.apply = patched_function_apply(Unpooling2D.apply)
else:
    Unpooling2D.__call__ = patched_function_call(Unpooling2D.__call__)

def to_mlir_node(self):
    return {
        b'name': 'Unpooling2D',
        b'params': {
            b'kh': self.kh,
            b'kw': self.kw,
            b'sy': self.sy,
            b'sx': self.sx,
            b'ph': self.ph,
            b'pw': self.pw,
            b'cover_all': self.cover_all,
            b'outh': self.outh,
            b'outw': self.outw,
        }
    }
Unpooling2D.to_mlir_node = to_mlir_node
