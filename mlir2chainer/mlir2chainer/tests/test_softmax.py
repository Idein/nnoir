import mlir
import chainer
import mlir2chainer
import numpy as np
import util

def test_softmax():
    inputs  = [mlir.Node('v0', 'float', (10,10))]
    outputs = [mlir.Node('v2', 'float', (10,10))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.Softmax(input_names, output_names, axis=1)
    result = mlir.MLIR('Softmax', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('softmax.mlir')

    x = np.random.randn(10,10)
    ref = function.run(x)
    m = mlir2chainer.ChainerNN('softmax.mlir')
    with chainer.using_config('train', False):
        y = m(x)
        assert(np.all(abs(y-ref).data<util.epsilon))
