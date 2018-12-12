import mlir
import chainer
import mlir2chainer
import numpy as np
import util

def test_reshape():
    inputs  = [mlir.Node('v0', 'float', (10,10))]
    outputs = [mlir.Node('v2', 'float', (5,20))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.Reshape(input_names, output_names, shape=(5,20))
    result = mlir.MLIR('Reshape', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('reshape.mlir')

    x = np.random.randn(5,20)
    ref = function.run(x)
    m = mlir2chainer.ChainerNN('reshape.mlir')
    with chainer.using_config('train', False):
        y = m(x)
        assert(np.all(abs(y-ref).data<util.epsilon))
