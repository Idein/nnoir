import mlir
import chainer
from mlir_chainer import MLIRFunction
import numpy as np
import util

def test_pad():
    inputs  = [mlir.Value(b'v0', 'float', (2,3,4,5))]
    outputs = [mlir.Value(b'v2', 'float', (3,5,7,10))]
    pads = ((1,0), (1,1), (1,2), (0,5))
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.functions.ConstantPadding(input_names, output_names, pads=pads, value=1)
    result = mlir.MLIR(b'ConstantPadding', b'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('pad.mlir')

    x = np.random.randn(2,3,4,5)
    ref = function.run(x)
    m = MLIRFunction('pad.mlir')
    with chainer.using_config('train', False):
        y = m(x)
        assert(np.all(abs(y-ref).data<util.epsilon))
