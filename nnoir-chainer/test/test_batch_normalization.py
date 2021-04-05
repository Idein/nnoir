import chainer
import chainer.links as L
import nnoir
import numpy as np
import util
from nnoir_chainer import NNOIRFunction


def test_batch_normalization():
    shape = (2, 3, 4, 5)
    channel = 3
    gamma = np.zeros(channel, dtype=np.float32)
    beta = np.zeros(channel, dtype=np.float32)
    avg_mean = np.zeros(channel, dtype=np.float32)
    avg_var = np.zeros(channel, dtype=np.float32)
    eps = 2e-05
    gamma[:] = 0.9
    beta[:] = 0.1
    avg_mean[:] = 0.2
    avg_var[:] = 0.8

    inputs = [nnoir.Value(b"v0", np.zeros(shape).astype("float32"))]
    outputs = [nnoir.Value(b"v2", np.zeros(shape).astype("float32"))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = nnoir.functions.BatchNormalization(
        input_names,
        output_names,
        eps=eps,
        avg_mean=avg_mean,
        avg_var=avg_var,
        gamma=gamma,
        beta=beta,
    )
    result = nnoir.NNOIR(
        b"BatchNormalization",
        b"nnoir2chainer_test",
        "0.1",
        input_names,
        output_names,
        nodes,
        [function],
    )
    result.dump("batch_normalization.nnoir")

    x = np.random.randn(*shape).astype("float32")
    ref = function.run(x)
    with chainer.using_config("train", False):
        m = NNOIRFunction("batch_normalization.nnoir")
        y = m(x)
        assert np.all(abs(y - ref).data < util.epsilon)
