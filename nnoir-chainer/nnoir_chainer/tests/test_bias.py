import nnoir
import chainer
from nnoir_chainer import NNOIRFunction
import numpy as np
import util


def test_bias():
    inputs = [nnoir.Value(b'v0', np.zeros((2, 3, 4, 5)).astype('float32'))]
    outputs = [nnoir.Value(b'v2', np.zeros((2, 3, 4, 5)).astype('float32'))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    b = np.random.randn(3).astype('float32')
    params = {'axis': 1,
              'b': b}
    function = nnoir.functions.Bias(input_names, output_names, **params)
    result = nnoir.NNOIR(b'Bias', b'nnoir2chainer_test', '0.1', input_names, output_names, nodes, [function])
    result.dump('bias.nnoir')

    x = np.random.randn(2, 3, 4, 5).astype('float32')
    ref = function.run(x)
    with chainer.using_config('train', False):
        m = NNOIRFunction('bias.nnoir')
        y = m(x)
        assert(np.all(abs(y-ref).data < util.epsilon))
