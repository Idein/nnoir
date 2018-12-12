import mlir
import chainer
import mlir2chainer
import numpy as np
import six
import msgpack
import util

def test_scale():
    inputs  = [mlir.Node('v0', 'float', (2,3,4,5))]
    outputs = [mlir.Node('v2', 'float', (2,3,4,5))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    W = np.random.randn(10)
    b = np.random.randn(10)
    params={'axis': 1,
            'W': W,
            'bias.b' : b}
    function = mlir.edges.Scale(input_names, output_names, **params)
    result = mlir.MLIR('Scale', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('scale.mlir')
    
    x = np.random.randn(10,10)
    ref = function.run(x)
    m = mlir2chainer.ChainerNN('scale.mlir')
    with chainer.using_config('train', False):
        y = m(x)
        assert(np.all(abs(y-ref).data<util.epsilon))

