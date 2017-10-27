import chainer.variable as variable

class Link(object):
    def __init__(self, chainer_base_link):
        self.chainer_base_link = chainer_base_link
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
