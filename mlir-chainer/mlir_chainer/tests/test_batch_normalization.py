import mlir
import chainer
from mlir_chainer import MLIRFunction
import numpy as np
import util
import chainer.links as L


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

    inputs = [mlir.Value(b'v0', np.zeros(shape).astype('float32'))]
    outputs = [mlir.Value(b'v2', np.zeros(shape).astype('float32'))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = mlir.functions.BatchNormalization(input_names, output_names,
                                                 eps=eps,
                                                 avg_mean=avg_mean,
                                                 avg_var=avg_var,
                                                 gamma=gamma,
                                                 beta=beta)
    result = mlir.MLIR(b'BatchNormalization', b'mlir2chainer_test', '0.1', input_names, output_names, nodes, [function])
    result.dump('batch_normalization.mlir')

    x = np.random.randn(*shape).astype('float32')
    ref = function.run(x)
    with chainer.using_config('train', False):
        m = MLIRFunction('batch_normalization.mlir')
        y = m(x)
        assert(np.all(abs(y-ref).data < util.epsilon))
