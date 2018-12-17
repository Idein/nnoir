import msgpack
import chainer
import sys
from .converter import *
from mlir.load import load

class MLIRFunction(chainer.Chain):
    def __init__(self, mlir_file):
        super(MLIRFunction, self).__init__()
        self.mlir = load(mlir_file)
        self._init_edges()

    def _init_edges(self):
        self.edges = {}
        nodes = { n.name: n for n in self.mlir.nodes }
        for edge in self.mlir.edges:
            inputs = [ nodes[x] for x in edge.inputs ]
            outputs = [ nodes[x] for x in edge.outputs ]
            self.edges[id(edge)] = globals()["Convert" + edge.__class__.__name__](edge, inputs, outputs)

    def __call__(self, *xs):
        if len(self.mlir.inputs) != len(xs):
            raise Exception("the number of input variables and the expected number of inputs do not match.")
        variable_dict = { inp:x for inp,x in zip(self.mlir.inputs, xs) }
        done = set()
        while len(done) < len(self.mlir.edges):
            found = True
            for edge in self.mlir.edges:
                if id(edge) not in done and set(edge.inputs).issubset(variable_dict.keys()):
                    variable_dict[edge.outputs[0]] = self.edges[id(edge)](*[ variable_dict[x] for x in edge.inputs ])
                    done.add(id(edge))
                    found = True
                    break
            if not found:
                raise Exception("invalid mlir.")
        ret = tuple( variable_dict[x] for x in self.mlir.outputs )
        if len(ret) == 1:
            return ret[0]
        else:
            return ret
