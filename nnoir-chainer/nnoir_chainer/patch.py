import numpy
import io
import inspect
from chainer.link import Link
from chainer.variable import Variable


def encode_ndarray(obj):
    x = None
    with io.BytesIO() as out:
        numpy.save(out, obj.copy())
        x = out.getvalue()
    return {b'ndarray': x}


def patched_link_call(orig_link_call):
    def call(self, *inputs, **d):
        self.chainer_input_variables = list(inputs)
        outputs = orig_link_call(self, *inputs, **d)
        if isinstance(outputs, Variable):
            self.chainer_output_variables = [outputs]
        else:
            self.chainer_output_variables = list(outputs)
        return outputs
    return call


def patched_function_apply(orig_function_apply):
    def apply(self, inputs):
        if isinstance(inputs, Variable):
            self.chainer_input_variables = [inputs]
        else:
            self.chainer_input_variables = list(inputs)
        outputs = orig_function_apply(self, inputs)
        if isinstance(outputs, Variable):
            self.chainer_output_variables = [outputs]
        else:
            self.chainer_output_variables = list(outputs)
        self.caller = caller_link()
        return outputs
    return apply


def patched_function_call(orig_function_call):
    def call(self, *inputs, **d):
        if isinstance(inputs, Variable):
            self.chainer_input_variables = [inputs]
        else:
            self.chainer_input_variables = list(inputs)
        outputs = orig_function_call(self, *inputs, **d)
        if isinstance(outputs, Variable):
            self.chainer_output_variables = [outputs]
        else:
            self.chainer_output_variables = list(outputs)
        self.caller = caller_link()
        return outputs
    return call


def caller_link():
    framerecords = inspect.stack()
    for framerecord in reversed(framerecords):
        frame = framerecord[0]
        arginfo = inspect.getargvalues(frame)
        result = arginfo.locals['self'] if 'self' in arginfo.locals else None
        if isinstance(result, Link) and hasattr(result, 'to_nnoir_node'):
            return result
    return None
