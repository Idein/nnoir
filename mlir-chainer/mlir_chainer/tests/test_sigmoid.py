import mlir
import chainer
from mlir_chainer import MLIRFunction
import numpy as np
import util


def test_sigmoid():
    inputs  = [mlir.Value(b'v0', np.zeros((10, 10)).astype('float32'))]
    outputs = [mlir.Value(b'v2', np.zeros((10, 10)).astype('float32'))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = mlir.functions.Sigmoid(input_names, output_names)
    result = mlir.MLIR(b'Sigmoid', b'mlir2chainer_test', '0.1', input_names, output_names, nodes, [function])
    result.dump('sigmoid.mlir')

    x = np.random.randn(10, 10).astype('float32')
    ref = function.run(x)
    with chainer.using_config('train', False):
        m = MLIRFunction('sigmoid.mlir')
        y = m(x)
        assert(np.all(abs(y-ref).data < util.epsilon))
