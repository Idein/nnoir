import mlir
import chainer
import mlir2chainer
import numpy as np
import six
import msgpack
import util
import chainer.links as L

def test_batch_normalization():
    shape = (2,3,4,5)
    channel = 3
    gamma = np.zeros(channel).astype(np.float32)
    beta = np.zeros(channel).astype(np.float32)
    avg_mean = np.zeros(channel).astype(np.float32)
    avg_var = np.zeros(channel).astype(np.float32)
    eps = 2e-05
    gamma[:] = 0.9
    beta[:] = 0.1
    avg_mean[:] = 0.2
    avg_var[:] = 0.8

    inputs  = [mlir.Node('v0', 'float', shape)]
    outputs = [mlir.Node('v2', 'float', shape)]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.BatchNormalization(input_names, output_names,
                                             eps=eps,
                                             avg_mean=avg_mean,
                                             avg_var=avg_var,
                                             gamma=gamma,
                                             beta=beta)
    result = mlir.MLIR('BatchNormalization', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('batch_normalization.mlir')
    
    x1 = np.random.randn(2,3,4,5).astype(np.float32)
    ref = function.run(x1)
    with chainer.using_config('train', False):
        m = mlir2chainer.ChainerNN('batch_normalization.mlir')
        y = m(x1)
        assert(np.all(abs(y-ref).data<util.epsilon))
