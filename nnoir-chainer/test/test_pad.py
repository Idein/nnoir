import chainer
import nnoir
import numpy as np
import util
from nnoir_chainer import NNOIRFunction


def test_pad():
    inputs = [nnoir.Value(b"v0", np.zeros((2, 3, 4, 5)).astype("float32"))]
    outputs = [nnoir.Value(b"v2", np.zeros((3, 5, 7, 10)).astype("float32"))]
    pads = ((1, 0), (1, 1), (1, 2), (0, 5))
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = nnoir.functions.ConstantPadding(input_names, output_names, pads=pads, value=1.0)
    result = nnoir.NNOIR(
        b"ConstantPadding",
        b"nnoir2chainer_test",
        "0.1",
        input_names,
        output_names,
        nodes,
        [function],
    )
    result.dump("pad.nnoir")

    x = np.random.randn(2, 3, 4, 5).astype("float32")
    ref = function.run(x)
    with chainer.using_config("train", False):
        m = NNOIRFunction("pad.nnoir")
        y = m(x)
        assert np.all(abs(y - ref).data < util.epsilon)
