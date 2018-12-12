import mlir
import chainer
import mlir2chainer
import numpy as np
import six
import msgpack
import util

def test_transpose():
    inputs  = [mlir.Node('v0', 'float', (2,3,4,5))]
    outputs = [mlir.Node('v2', 'float', (4,5,2,3))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.Transpose(input_names, output_names, axes=(2,3,0,1))
    result = mlir.MLIR('Transpose', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('transpose.mlir')

    x = np.random.randn(2,3,4,5)
    ref = function.run(x)
    m = mlir2chainer.ChainerNN('transpose.mlir')
    with chainer.using_config('train', False):
        y = m(x)
        assert(np.all(abs(y-ref).data<util.epsilon))
