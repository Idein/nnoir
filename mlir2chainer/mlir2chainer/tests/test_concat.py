import mlir
import chainer
import mlir2chainer
import numpy as np
import util

def test_concat():
    inputs  = [mlir.Node('v0', 'float', (10,11)),
               mlir.Node('v1', 'float', (10,12))]
    outputs = [mlir.Node('v2', 'float', (10,10))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.Concat(input_names, output_names, axis=1)
    result = mlir.MLIR('Concat', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('concat.mlir')

    x1 = np.random.randn(10,11)
    x2 = np.random.randn(10,12)
    ref = function.run(x1, x2)
    m = mlir2chainer.ChainerNN('concat.mlir')
    with chainer.using_config('train', False):
        y = m(x1, x2)
        assert(np.all(abs(y-ref).data<util.epsilon))
