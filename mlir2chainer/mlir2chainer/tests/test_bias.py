import mlir
import chainer
import mlir2chainer
import numpy as np
import util

def test_scale():
    inputs  = [mlir.Node('v0', 'float', (2,3,4,5))]
    outputs = [mlir.Node('v2', 'float', (2,3,4,5))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    b = np.random.randn(10)
    params={'axis': 1,
            'b' : b}
    function = mlir.edges.Bias(input_names, output_names, **params)
    result = mlir.MLIR('Bias', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('bias.mlir')

    x = np.random.randn(10,10)
    ref = function.run(x)
    m = mlir2chainer.ChainerNN('bias.mlir')
    with chainer.using_config('train', False):
        y = m(x)
        assert(np.all(abs(y-ref).data<util.epsilon))
