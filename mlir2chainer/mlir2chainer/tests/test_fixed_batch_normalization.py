import mlir
import chainer
import mlir2chainer
import numpy as np
import six
import msgpack
import util
import chainer.links as L

def test_fixed_batch_normalization():
    shape = (2,3,4,5)
    channel = 3
    gamma = np.zeros(channel)
    beta = np.zeros(channel)
    avg_mean = np.zeros(channel)
    avg_var = np.zeros(channel)
    eps = 2e-05
    gamma[:] = 0.9
    beta[:] = 0.1
    avg_mean[:] = 0.2
    avg_var[:] = 0.8

    inputs  = [mlir.Node('v0', 'float', shape),
               mlir.Node('v1', 'float', (channel,)),
               mlir.Node('v2', 'float', (channel,)),
               mlir.Node('v3', 'float', (channel,)),
               mlir.Node('v4', 'float', (channel,))]
    outputs = [mlir.Node('y', 'float', shape)]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.FixedBatchNormalization(input_names, output_names, eps=eps)
    result = mlir.MLIR('FixedBatchNormalization', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('fixed_batch_normalization.mlir')
    
    x = np.random.randn(2,3,4,5)
    ref = function.run(x, gamma, beta, avg_mean, avg_var)
    with chainer.using_config('train', False):
        m = mlir2chainer.ChainerNN('fixed_batch_normalization.mlir')
        y = m(x, gamma, beta, avg_mean, avg_var)
        assert(np.all(abs(y-ref).data<util.epsilon))
