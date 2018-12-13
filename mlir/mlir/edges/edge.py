import numpy
import six

class Edge(object):
    def __init__(self, inputs, outputs, params, necessary_params, optional_params):
        if necessary_params - set(params.keys()) != set():
            lacks = ", ".join(necessary_params - set(params.keys()))
            raise Exception("lack of necessary parameter: {}".format(lacks))
        if set(params.keys()) - necessary_params - optional_params != set():
            unknowns = ", ".join(set(params.keys()) - necessary_params - optional_params)
            raise Exception("unknown parameter: {}".format(unknowns))
        self.inputs = inputs
        self.outputs = outputs
        self.params = params

    def dump(self):
        def encode_ndarray(obj):
            x = None
            with six.BytesIO() as out:
                numpy.save(out, obj.copy())
                x = out.getvalue()
            return { b'ndarray': x }
        binary_params = {}
        for k,v in self.params.items():
            if type(v) is numpy.ndarray:
                binary_params[k.encode()] = encode_ndarray(v)
            else:
                binary_params[k.encode()] = v
        return {
            b'name': self.__class__.__name__,
            b'inputs': self.inputs,
            b'outputs': self.outputs,
            b'params': binary_params
        }
