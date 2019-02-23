from chainer.functions.pooling.unpooling_2d import Unpooling2D
from nnoir_chainer.patch import patched_function_apply, patched_function_call
import nnoir.functions as NNOIR

if hasattr(Unpooling2D, 'apply'):
    Unpooling2D.apply = patched_function_apply(Unpooling2D.apply)
else:
    Unpooling2D.__call__ = patched_function_call(Unpooling2D.__call__)


def to_nnoir_node(self, inputs, outputs):
    return NNOIR.Unpooling2D(
        [x.name for x in inputs],
        [x.name for x in outputs],
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


Unpooling2D.to_nnoir_node = to_nnoir_node
