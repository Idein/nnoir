import chainer
import chainer.functions as functions
from chainer.mlir import node
from chainer.mlir.node import encode_ndarray

class Add(node.Function, functions.Add):
    def __init__(self, *inputs, **dicts):
        super(Add, self).__init__(functions.Add)
        super(node.Function, self).__init__(*inputs, **dicts)

    def to_mlir_node(self):
        return {
            b'name': self.chainer_node_label,
            b'params': {}
        }

def add(self, rhs):
    if isinstance(rhs, variable.Variable):
        return Add().apply((self, rhs))[0]
    else:
        pass
        # return AddConstant(rhs).apply((self,))[0]

variable.Variable.__add__ = add
