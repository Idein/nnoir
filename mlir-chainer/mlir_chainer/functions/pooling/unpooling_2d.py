from chainer.functions.pooling.unpooling_2d import Unpooling2D
from mlir_chainer.patch import patched_function_apply, patched_function_call
import mlir.functions as MLIR

if hasattr(Unpooling2D, 'apply'):
    Unpooling2D.apply = patched_function_apply(Unpooling2D.apply)
else:
    Unpooling2D.__call__ = patched_function_call(Unpooling2D.__call__)


def to_mlir_node(self, inputs, outputs):
    return MLIR.Unpooling2D(
        inputs,
        outputs,
        kh=self.kh,
        kw=self.kw,
        sy=self.sy,
        sx=self.sx,
        ph=self.ph,
        pw=self.pw,
        cover_all=self.cover_all,
        outh=self.outh,
        outw=self.outw,
    )


Unpooling2D.to_mlir_node = to_mlir_node
