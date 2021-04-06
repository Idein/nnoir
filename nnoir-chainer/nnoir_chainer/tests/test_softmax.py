import chainer
import nnoir
import numpy as np
import util
from nnoir_chainer import NNOIRFunction


def test_softmax():
    inputs = [nnoir.Value(b"v0", np.zeros((10, 10)).astype("float32"))]
    outputs = [nnoir.Value(b"v2", np.zeros((10, 10)).astype("float32"))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = nnoir.functions.Softmax(input_names, output_names, axis=1)
    result = nnoir.NNOIR(
        b"Softmax",
        b"nnoir2chainer_test",
        "0.1",
        input_names,
        output_names,
        nodes,
        [function],
    )
    result.dump("softmax.nnoir")

    x = np.random.randn(10, 10).astype("float32")
    ref = function.run(x)
    with chainer.using_config("train", False):
        m = NNOIRFunction("softmax.nnoir")
        y = m(x)
        assert np.all(abs(y - ref).data < util.epsilon)
