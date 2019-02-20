import nnoir
import chainer
from nnoir_chainer import NNOIRFunction
import numpy as np
import util


def test_add():
    inputs = [nnoir.Value(b'v0', np.zeros((10, 10)).astype('float32')),
              nnoir.Value(b'v1', np.zeros((10, 10)).astype('float32'))]
    outputs = [nnoir.Value(b'v2', np.zeros((10, 10)).astype('float32'))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = nnoir.functions.Add(input_names, output_names)
    result = nnoir.NNOIR(b'Add', b'nnoir2chainer_test', '0.1', input_names, output_names, nodes, [function])
    result.dump('add.nnoir')

    x1 = np.random.randn(10, 10).astype('float32')
    x2 = np.random.randn(10, 10).astype('float32')
    ref = function.run(x1, x2)
    with chainer.using_config('train', False):
        m = NNOIRFunction('add.nnoir')
        y = m(x1, x2)
        assert(np.all(abs(y-ref).data < util.epsilon))
