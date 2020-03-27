import nnoir
import chainer
from nnoir_chainer import NNOIRFunction
import numpy as np
import util


def test_concat():
    v0 = nnoir.Value(b'v0', np.zeros((10, 11)).astype('float32'))
    v1 = nnoir.Value(b'v1', np.zeros((10, 12)).astype('float32'))
    graph_inputs = [v0, v1]
    inputs = [v0,
              v0,
              v1]
    outputs = [nnoir.Value(b'v2', np.zeros((10, 34)).astype('float32'))]
    graph_input_names = [x.name for x in graph_inputs]
    nodes = graph_inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    function = nnoir.functions.Concat(input_names, output_names, axis=1)
    result = nnoir.NNOIR(b'Concat', b'nnoir2chainer_test', '0.1', graph_input_names, output_names, nodes, [function])
    result.dump('concat.nnoir')

    x1 = np.random.randn(10, 11).astype('float32')
    x2 = np.random.randn(10, 12).astype('float32')
    ref = function.run(x1, x1, x2)
    with chainer.using_config('train', False):
        m = NNOIRFunction('concat.nnoir')
        y = m(x1, x2)
        assert(np.all(abs(y-ref).data < util.epsilon))
