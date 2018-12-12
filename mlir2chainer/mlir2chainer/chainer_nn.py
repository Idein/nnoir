import msgpack
import chainer
import sys
from .converter import *
from mlir.load import load

class ChainerNN(chainer.Chain):
    def __init__(self, mlir_file):
        self.mlir = load(mlir_file)
        super(ChainerNN, self).__init__()

    def __call__(self, *xs):
        if len(self.mlir.inputs) != len(xs):
            raise Exception("the number of input variables and the expected number of inputs do not match.")
        variable_dict = { inp:x for inp,x in zip(self.mlir.inputs, xs) }
        for edge in self.mlir.edges:
            converter = globals()["Convert" + edge.__class__.__name__]()
            output_from = converter.to_chainer(edge, *[ variable_dict[x] for x in edge.inputs ])
            variable_dict[edge.outputs[0]] = output_from
        ret = tuple( variable_dict[x] for x in self.mlir.outputs )
        if len(ret) == 1:
            return ret[0]
        else:
            return ret


