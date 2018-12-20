import mlir
import chainer
from mlir_chainer import MLIRFunction
import numpy as np
import util

def test_concat():
    inputs  = [mlir.Node(b'v0', 'float', (10,11)),
               mlir.Node(b'v1', 'float', (10,12))]
    outputs = [mlir.Node(b'v2', 'float', (10,10))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.Concat(input_names, output_names, axis=1)
    result = mlir.MLIR(b'Concat', b'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('concat.mlir')

    x1 = np.random.randn(10,11)
    x2 = np.random.randn(10,12)
    ref = function.run(x1, x2)
    m = MLIRFunction('concat.mlir')
    with chainer.using_config('train', False):
        y = m(x1, x2)
        assert(np.all(abs(y-ref).data<util.epsilon))
