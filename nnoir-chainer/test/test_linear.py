import nnoir
import chainer
from nnoir_chainer import NNOIRFunction
import numpy as np
import util


def test_linear():
    batch = 2
    in_ch = 3
    out_ch = 4
    inputs = [nnoir.Value(b'v0', np.zeros((batch, in_ch)).astype('float32'))]
    outputs = [nnoir.Value(b'v2', np.zeros((batch, out_ch)).astype('float32'))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    W = np.random.randn(out_ch, in_ch).astype('float32')
    b = np.random.randn(out_ch).astype('float32')
    function = nnoir.functions.Linear(input_names, output_names, W=W, b=b)
    result = nnoir.NNOIR(b'Linear', b'nnoir2chainer_test', '0.1', input_names, output_names, nodes, [function])
    result.dump('linear.nnoir')

    x = np.random.randn(batch, in_ch).astype('float32')
    ref = function.run(x)
    with chainer.using_config('train', False):
        m = NNOIRFunction('linear.nnoir')
        y = m(x)
        assert(np.all(abs(y-ref).data < util.epsilon))
