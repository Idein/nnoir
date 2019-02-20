import nnoir
import chainer
from nnoir_chainer import NNOIRFunction
import numpy as np
import util


def test_transpose():
    inputs = [nnoir.Value(b'v0', np.zeros((2, 3, 4, 5)).astype('float32'))]
    outputs = [nnoir.Value(b'v2', np.zeros((4, 5, 2, 3)).astype('float32'))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = nnoir.functions.Transpose(input_names, output_names, axes=(2, 3, 0, 1))
    result = nnoir.NNOIR(b'Transpose', b'nnoir2chainer_test', '0.1', input_names, output_names, nodes, [function])
    result.dump('transpose.nnoir')

    x = np.random.randn(2, 3, 4, 5).astype('float32')
    ref = function.run(x)
    with chainer.using_config('train', False):
        m = NNOIRFunction('transpose.nnoir')
        y = m(x)
        assert(np.all(abs(y-ref).data < util.epsilon))
