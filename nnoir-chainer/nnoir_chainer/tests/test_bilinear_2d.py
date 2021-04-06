import sys

import chainer
import chainer.functions as F
import nnoir
import numpy as np
import util
from nnoir_chainer import NNOIRFunction


def test_bilinear_2d_without_mode():
    in_shape = (2, 3, 9, 10)
    out_shape = (2, 3, 4, 5)
    inputs = [nnoir.Value(b"v0", np.zeros(in_shape).astype("float32"))]
    outputs = [nnoir.Value(b"v2", np.zeros(out_shape).astype("float32"))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = nnoir.functions.Bilinear2D(input_names, output_names, size=(out_shape[2], out_shape[3]))

    result = nnoir.NNOIR(
        b"Bilinear2D",
        b"nnoir2chainer_test",
        "0.1",
        input_names,
        output_names,
        nodes,
        [function],
    )
    result.dump("bilinear_2d_without_mode.nnoir")

    x = np.random.randn(*in_shape).astype("float32")
    ref = function.run(x)
    with chainer.using_config("train", False):
        m = NNOIRFunction("bilinear_2d_without_mode.nnoir")
        y = m(x)
        assert np.all(abs(y - ref).data < util.epsilon)


def test_bilinear_2d_with_mode():
    in_shape = (2, 3, 9, 10)
    out_shape = (2, 3, 4, 5)
    inputs = [nnoir.Value(b"v0", np.zeros(in_shape).astype("float32"))]
    outputs = [nnoir.Value(b"v2", np.zeros(out_shape).astype("float32"))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = nnoir.functions.Bilinear2D(
        input_names,
        output_names,
        size=(out_shape[2], out_shape[3]),
        mode=b"align_corners",
    )

    result = nnoir.NNOIR(
        b"Bilinear2D",
        b"nnoir2chainer_test",
        "0.1",
        input_names,
        output_names,
        nodes,
        [function],
    )
    result.dump("bilinear_2d_with_mode.nnoir")

    x = np.random.randn(*in_shape).astype("float32")
    ref = function.run(x)
    with chainer.using_config("train", False):
        m = NNOIRFunction("bilinear_2d_with_mode.nnoir")
        y = m(x)
        assert np.all(abs(y - ref).data < util.epsilon)
