import numpy
import inspect
import chainer.variable as variable
import six

class Node(object):
    def to_mlir_node(self):
        pass

def encode_ndarray(obj):
    x = None
    with six.BytesIO() as out:
        numpy.save(out, obj.copy())
        x = out.getvalue()
    return { b'ndarray': x }

class Link(Node):
    def __init__(self, chainer_base_link):
        self.chainer_base_link = chainer_base_link
        self.chainer_node_label = chainer_base_link.__name__
        self.chainer_input_variables = None
        self.chainer_output_variables = None

    def __call__(self, *inputs, **d):
        self.chainer_input_variables = list(inputs)
        outputs = self.chainer_base_link.__call__(self, *inputs, **d)
        if isinstance(outputs, variable.Variable):
            self.chainer_output_variables = [outputs]
        else:
            self.chainer_output_variables = list(outputs)
        return outputs

    def to_mlir_node(self):
        pass

def caller_link():
    framerecords = inspect.stack()
    for framerecord in framerecords:
        frame = framerecord[0]
        arginfo = inspect.getargvalues(frame)
        result = arginfo.locals['self'] if 'self' in arginfo.locals else None
        if isinstance(result, Node):
            return result
    return None

class Function(Node):
    def __init__(self, chainer_base_function):
        self.chainer_base_function = chainer_base_function
        self.chainer_node_label = chainer_base_function.__name__
        self.chainer_input_variables = None
        self.chainer_output_variables = None
        self.caller = None

    def apply(self, inputs):
        if hasattr(self.chainer_base_function, 'apply'):
            if isinstance(inputs, variable.Variable):
                self.chainer_input_variables = [inputs]
            else:
                self.chainer_input_variables = list(inputs)
            outputs = self.chainer_base_function.apply(self, inputs)
            if isinstance(outputs, variable.Variable):
                self.chainer_output_variables = [outputs]
            else:
                self.chainer_output_variables = list(outputs)
            self.caller = caller_link()
            return outputs
        else:
            # TODO: throw Error
            pass

    def __call__(self, inputs, **d):
        if hasattr(self.chainer_base_function, 'apply'):
            # TODO: throw Error
            pass
        else:
            if isinstance(inputs, variable.Variable):
                self.input_variables = [inputs]
            else:
                self.input_variables = list(inputs)
            outputs = self.chainer_base_function.__call__(self, inputs, **d)
            if isinstance(outputs, variable.Variable):
                self.output_variables = [outputs]
            else:
                self.output_variables = list(outputs)
            self.caller = caller_link()
            return outputs
