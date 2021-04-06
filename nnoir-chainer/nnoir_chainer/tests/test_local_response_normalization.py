import chainer
import nnoir
import numpy as np
import util
from nnoir_chainer import NNOIRFunction


def test_local_response_normalization():
    inputs = [nnoir.Value(b"v0", np.zeros((10, 10)).astype("float32"))]
    outputs = [nnoir.Value(b"v2", np.zeros((10, 10)).astype("float32"))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = nnoir.functions.LocalResponseNormalization(input_names, output_names, n=6, k=3.0, alpha=0.0002, beta=0.8)
    result = nnoir.NNOIR(
        b"LocalResponseNormalization",
        b"nnoir2chainer_test",
        "0.1",
        input_names,
        output_names,
        nodes,
        [function],
    )
    result.dump("local_response_normalization.nnoir")

    x = np.random.randn(10, 10).astype("float32")
    ref = function.run(x)
    with chainer.using_config("train", False):
        m = NNOIRFunction("local_response_normalization.nnoir")
        y = m(x)
        assert np.all(abs(y - ref).data < util.epsilon)
