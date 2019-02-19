import mlir
import chainer
from mlir_chainer import MLIRFunction
import numpy as np
import util


def test_transpose():
    inputs  = [mlir.Value(b'v0', np.zeros((2, 3, 4, 5)).astype('float32'))]
    outputs = [mlir.Value(b'v2', np.zeros((4, 5, 2, 3)).astype('float32'))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = mlir.functions.Transpose(input_names, output_names, axes=(2, 3, 0, 1))
    result = mlir.MLIR(b'Transpose', b'mlir2chainer_test', '0.1', input_names, output_names, nodes, [function])
    result.dump('transpose.mlir')

    x = np.random.randn(2, 3, 4, 5).astype('float32')
    ref = function.run(x)
    with chainer.using_config('train', False):
        m = MLIRFunction('transpose.mlir')
        y = m(x)
        assert(np.all(abs(y-ref).data < util.epsilon))
