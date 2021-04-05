import chainer
import chainer.functions as F
import nnoir
import numpy as np
import util
from nnoir_chainer import NNOIRFunction


def test_average_pooling_2d():
    ksize = (2, 3)
    stride = (1, 2)
    pad = (0, 1)

    inputs = [nnoir.Value(b"v0", np.zeros((2, 3, 4, 5)).astype("float32"))]
    outputs = [nnoir.Value(b"v2", np.zeros((2, 3, 3, 3)).astype("float32"))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = nnoir.functions.AveragePooling2D(
        input_names,
        output_names,
        kernel=list(ksize),
        stride=list(stride),
        pad_h=[pad[0], pad[0] + stride[0] - 1],
        pad_w=[pad[1], pad[1] + stride[1] - 1],
        count_exclude_pad=False,
    )
    result = nnoir.NNOIR(
        b"AveragePooling2D",
        b"nnoir2chainer_test",
        "0.1",
        input_names,
        output_names,
        nodes,
        [function],
    )
    result.dump("average_pooling_2d.nnoir")

    x = np.random.randn(2, 3, 4, 5).astype("float32")
    ref = function.run(x)
    with chainer.using_config("train", False):
        m = NNOIRFunction("average_pooling_2d.nnoir")
        y = m(x)
        assert np.all(abs(y - ref).data < util.epsilon)
