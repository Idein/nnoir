import mlir
import chainer
from mlir_chainer import MLIRFunction
import numpy as np
import util


def test_mul():
    inputs  = [mlir.Value(b'v0', np.zeros((10, 10)).astype('float32')),
               mlir.Value(b'v1', np.zeros((10, 10)).astype('float32'))]
    outputs = [mlir.Value(b'v2', np.zeros((10, 10)).astype('float32'))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = mlir.functions.Mul(input_names, output_names)
    result = mlir.MLIR(b'Mul', b'mlir2chainer_test', '0.1', input_names, output_names, nodes, [function])
    result.dump('mul.mlir')

    x1 = np.random.randn(10, 10).astype('float32') 
    x2 = np.random.randn(10, 10).astype('float32') 
    ref = function.run(x1, x2)
    with chainer.using_config('train', False):
        m = MLIRFunction('mul.mlir')
        y = m(x1, x2)
        assert(np.all(abs(y-ref) < util.epsilon))
