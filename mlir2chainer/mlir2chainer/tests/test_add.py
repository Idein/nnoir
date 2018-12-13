import mlir
import chainer
import mlir2chainer
import numpy as np
import util

def test_add():
    inputs  = [mlir.Node('v0', 'float', (10,10)),
               mlir.Node('v1', 'float', (10,10))]
    outputs = [mlir.Node('v2', 'float', (10,10))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.Add(input_names, output_names)
    result = mlir.MLIR('Add', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('add.mlir')
    
    x1 = np.random.randn(10,10)
    x2 = np.random.randn(10,10)
    ref = function.run(x1, x2)
    m = mlir2chainer.ChainerNN('add.mlir')
    with chainer.using_config('train', False):
        y = m(x1, x2)
        assert(np.all(abs(y-ref).data<util.epsilon))
