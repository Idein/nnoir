import mlir
import chainer
from mlir_chainer import MLIRFunction
import numpy as np
import sys
import chainer.functions as F
import util


def test_broadcast():
    in_shape = (1, 1, 4, 5)
    out_shape = (2, 3, 4, 5)
    inputs  = [mlir.Value(b'v0', np.zeros(in_shape).astype('float32'))]
    outputs = [mlir.Value(b'v2', np.zeros(out_shape).astype('float32'))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = mlir.functions.BroadcastTo(input_names, output_names)

    result = mlir.MLIR(b'BroadcastTo', b'mlir2chainer_test', '0.1', input_names, output_names, nodes, [function])
    result.dump('broadcast.mlir')

    x = np.random.randn(*in_shape).astype('float32')
    ref = function.run(x, out_shape)
    with chainer.using_config('train', False):
        m = MLIRFunction('broadcast.mlir')
        y = m(x)
        assert(np.all(abs(y-ref).data < util.epsilon))
