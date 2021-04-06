import chainer
import nnoir
import numpy as np
import util
from nnoir_chainer import NNOIRFunction


def test_unpooling_2d():
    kh, kw = 2, 3
    sy, sx = 1, 2
    batch = 2
    ch = 3
    in_h, in_w = 5, 6
    out_h, out_w = 4, 9
    ph, pw = 1, 2
    inputs = [nnoir.Value(b"v0", np.zeros((batch, ch, in_h, in_w)).astype("float32"))]
    outputs = [nnoir.Value(b"v2", np.zeros((batch, ch, out_h, out_w)).astype("float32"))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = nnoir.functions.Unpooling2D(
        input_names,
        output_names,
        kh=kh,
        kw=kw,
        sy=sy,
        sx=sx,
        ph=ph,
        pw=pw,
        cover_all=False,
        outh=out_h,
        outw=out_w,
    )
    result = nnoir.NNOIR(
        b"Unpooling2D",
        b"nnoir2chainer_test",
        "0.1",
        input_names,
        output_names,
        nodes,
        [function],
    )
    result.dump("unpooling_2d.nnoir")

    x = np.random.randn(batch, ch, in_h, in_w).astype("float32")
    ref = function.run(x)
    with chainer.using_config("train", False):
        m = NNOIRFunction("unpooling_2d.nnoir")
        y = m(x)
        assert np.all(abs(y - ref).data < util.epsilon)
