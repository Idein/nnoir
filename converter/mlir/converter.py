# -*- coding: utf-8 -*-
import heapq
import inspect
import sys
import msgpack
import numpy

import chainer
from chainer import function
from chainer import variable
import chainer.links as L
import chainer.functions as F
import chainer.functions.math.basic_math as M

def encode_ndarray(obj):
    x = obj
    if sys.byteorder == 'little':
        if obj.dtype.byteorder != '>':
            x = obj.byteswap().newbyteorder('B')
    else:
        if obj.dtype.byteorder == '<':
            x = obj.byteswap().newbyteorder('B')
    return { b'ndarray':
             { b'dtype': x.dtype.str,
               b'shape': x.shape,
               b'data': x.tostring() } }

def decode(obj):
    x = obj[b'ndarray']
    dtype = numpy.dtype(x[b'dtype'])
    return numpy.fromstring(x[b'data'], dtype=dtype).reshape(x[b'shape'])

class Node(object):
    def __init__(self, node, no = None):
        self.no = no
        self.node = node
        self.in_edges = None
        self.out_edges = None
    def __hash__(self):
        return self.node.__hash__()
    def __eq__(self, r):
        return self.node is r.node
    def add_in_edge(self, from_node):
        if self.in_edges is None: self.in_edges = []
        self.in_edges.append(from_node)
    def add_out_edge(self, to_node):
        if self.out_edges is None: self.out_edges = []
        self.out_edges.append(to_node)
    def is_type_of(self, cls):
        return isinstance(self.node, cls)

additional_params = {
    'Convolution2D': ['W', 'b', 'stride', 'pad'],
    'Inception': [
        { 'conv1': ['W', 'b'] },
        { 'proj3': ['W', 'b'] },
        { 'conv3': ['W', 'b'] },
        { 'proj5': ['W', 'b'] },
        { 'conv5': ['W', 'b'] },
        { 'projp': ['W', 'b'] } ],
    'InceptionBN': [
        { 'proj3': ['W', 'b'] },
        { 'conv3': ['W', 'b'] },
        { 'proj33': ['W', 'b'] },
        { 'conv33a': ['W', 'b'] },
        { 'conv33b': ['W', 'b'] } ],
    'LSTM': [
        { 'upward': ['W', 'b'] },
        { 'lateral': ['W'] } ],
    'EmbedID': ['W'],
    'Linear': ['W', 'b'],
    'MulConstant': ['value'],
    'Convolution2DFunction': ['sy', 'sx', 'ph', 'pw', 'cover_all'],
    'SoftmaxCrossEntropy': ['normalize', 'cache_score'],
    'Dropout': ['dropout_ratio'],
    'AveragePooling2D': ['kh', 'kw', 'sy', 'sx', 'ph', 'pw', 'cover_all'],
    'MaxPooling2D': ['kh', 'kw', 'sy', 'sx', 'ph', 'pw', 'cover_all'],
    'Concat': ['axis'],
    'LocalResponseNormalization': ['n', 'k', 'alpha', 'beta'],
    'BatchNormalizationFunction': ['eps', 'running_mean', 'running_var', 'train', 'decay'],
    'LeakyReLU' : ['slope'],
}

def patch_for_links():
    # patch for Links (and Chains)
    target_links = []
    for k,v in L.__dict__.items():
        if inspect.isclass(v):
            target_links.append(v)
    orig_link_calls = {l.__name__: l.__call__ for l in target_links}
    def link_call(self, *inputs, **d):
        self.input_variables = list(inputs)
        outputs = orig_link_calls[self.__class__.__name__](self, *inputs, **d)
        if isinstance(outputs, variable.Variable):
            self.output_variables = [outputs]
        else:
            self.output_variables = list(outputs)
        return outputs
    for l in target_links:
        l.__call__ = link_call
        l.label = l.__name__

def patch_for_functions():
    # patch for Functions
    target_functions = []
    for k,v in (list(F.__dict__.items()) + list(M.__dict__.items())):
        if inspect.isclass(v):
            target_functions.append(v)
    orig_function_calls = {f.__name__: f.__call__ for f in target_functions}
    def function_call(self, *inputs, **d):
        self.input_variables = list(inputs)
        outputs = orig_function_calls[self.__class__.__name__](self, *inputs, **d)
        if isinstance(outputs, variable.Variable):
            self.output_variables = [outputs]
        else:
            self.output_variables = list(outputs)
        return outputs
    for f in target_functions:
        f.__call__ = function_call
        f.label = f.__name__

def patch():
    patch_for_links()
    patch_for_functions()

class Chainer(object):
    def __init__(self, model, input_variables, output_variables):
        self.input_variables = []
        self.output_variables = []
        self.nodes = set()
        self.model = model
        self.additional_params = additional_params

        push_count = [0]
        candidates = []
        def add_candidate(candidate):
            heapq.heappush(candidates, (push_count[0], candidate))
            push_count[0] += 1

        n2N = {}
        created_order = [0]
        def create_node(node):
            if node in n2N:
                return n2N[node]
            else:
                n2N[node] = Node(node, created_order[0])
                created_order[0] += 1
                return n2N[node]

        out2link = {}
        for child in model.children():
            if hasattr(child, 'output_variables'):
                out2link.update({id(o): child for o in child.output_variables})

        # Create graph
        seen_edges = set()
        for o in output_variables:
            add_candidate(o)
            o_node = create_node(o)
            self.nodes.add(o_node)
            self.output_variables.append(o_node)
        while candidates:
            _, candidate = heapq.heappop(candidates)
            if isinstance(candidate, variable.Variable):
                creator = None
                if id(candidate) in out2link:
                    creator = out2link[id(candidate)]
                else:
                    creator = candidate.creator
                if creator is not None and (creator, candidate) not in seen_edges:
                    add_candidate(creator)
                    seen_edges.add((creator, candidate))
                    creator_node = create_node(creator)
                    candidate_node = create_node(candidate)
                    candidate_node.add_in_edge(creator_node)
                    creator_node.add_out_edge(candidate_node)
                    self.nodes.add(creator_node)
            else:
                for input_ in reversed(candidate.input_variables):
                    if input_ is not candidate and (input_, candidate) not in seen_edges:
                        add_candidate(input_)
                        seen_edges.add((input_, candidate))
                        input_node = create_node(input_)
                        candidate_node = create_node(candidate)
                        candidate_node.add_in_edge(input_node)
                        input_node.add_out_edge(candidate_node)
                        self.nodes.add(input_node)
        for i in input_variables:
            self.input_variables.append(create_node(i))

        # Topological sorting
        visited = set()
        sorted_nodes = []
        def visit(node):
            if node not in visited:
                visited.add(node)
                if node.out_edges is not None:
                    for n in node.out_edges:
                        visit(n)
                sorted_nodes.insert(0, node)
        for node in sorted(self.nodes, key=lambda n: n.no):
            visit(node)
        for no, node in enumerate(sorted_nodes):
            node.no = no

    def to_hs(self):
        iargs = ' '.join(map(lambda t: 'input%d' % t[0], enumerate(self.input_variables)))
        oargs = ', '.join(map(lambda t: 'output%d' % t[0], enumerate(self.output_variables)))
        ret = "forward %s = (%s) where\n" % (iargs, oargs)
        for node in sorted(self.nodes, key=lambda n: n.no):
            if isinstance(node.node, variable.Variable):
                creator = node.node.creator
                if creator is None:
                    ret += "  %s = %s\n" % (self._lvalue_name(node), self._rvalue_name(node))
                else:
                    creator = node.in_edges[0]
                    args = ' '.join(map(lambda node: self._lvalue_name(node), creator.in_edges))
                    ret += "  %s = forward%s %s\n" % (self._lvalue_name(node), self._function_name(creator), args)
        return ret

    def _variable_elem_name(self, node):
        return "v%s%d" % ('' if node.node.name is None else node.node.name, node.no)

    def _variable_name(self, node):
        return "v%s%d %s" % ('' if node.node.name is None else node.node.name, node.no, node.node.shape)

    def _lvalue_name(self, node):
        ix = reduce(lambda e, t: t[0] if e is None and node is t[1] else e, enumerate(self.output_variables), None)
        if ix is None:
            return self._variable_name(node)
        else:
            return "output%d" % ix

    def _rvalue_name(self, node):
        ix = reduce(lambda e, t: t[0] if e is None and node is t[1] else e, enumerate(self.input_variables), None)
        if ix is None:
            return "[abbrev.]" # '"%s"'
        else:
            return "input%d" % ix

    def _function_name(self, node):
        aps = self.additional_params[node.node.label] if node.node.label in self.additional_params else None
        in_edges = '|'.join(map(lambda v: '<' + self._variable_elem_name(v) + '>', reversed(node.in_edges)))
        out_edges = '|'.join(map(lambda v: '<' + self._variable_elem_name(v) + '>', reversed(node.out_edges)))
        if aps is None:
            return "{{%s}|%s|{%s}}" % (in_edges, node.node.label, out_edges)
        else:
            def find_params(n, ps):
                for p in ps:
                    if isinstance(p, str):
                        yield str(n.__dict__[p])
                    else:
                        for k,vs in p.items():
                            for param in find_params(n.__dict__[k], vs):
                                yield "%s.%s" % (k, param)
            #return "%s %s" % (node.node.label, ' '.join(map(lambda p: str(node.node.__dict__[p]), aps)))
            return "{{%s}|%s %s|{%s}}" % (in_edges, node.node.label, ' '.join(find_params(node.node, aps)), out_edges)

    def to_dot(self, variable_style = None, function_style = None, rankdir = 'TB'):
        ret = 'digraph graphname { rankdir=%s;\n' % rankdir
        ret += '  subgraph input {\n'
        for i, node in enumerate(self.input_variables):
            attribute = { 'xlabel' : 'input%d = %s' % (i, self._variable_name(node)),
                          'shape' : 'point' }
            attributes = ["%s=\"%s\"" % (k, v) for (k, v)
                          in attribute.items()]
            ret += "    %s [%s];\n" % (id(node.node), ",".join(attributes))
        ret += '  }\n'
        ret += '  subgraph output {\n'
        for i, node in enumerate(self.output_variables):
            attribute = { 'xlabel' : 'output%d = %s' % (i, self._variable_name(node)),
                          'shape' : 'point' }
            attributes = ["%s=\"%s\"" % (k, v) for (k, v)
                          in attribute.items()]
            ret += "    %s [%s];\n" % (id(node.node), ",".join(attributes))
        ret += '  }\n'
        for node in self.nodes:
            if node in self.input_variables:
                continue
            if node in self.output_variables:
                continue
            # assert isinstance(node.node, (variable.Variable, function.Function))
            if isinstance(node.node, variable.Variable):
                attribute = { 'xlabel' : self._variable_name(node),
                              'shape' : 'point' }
                attributes = ["%s=\"%s\"" % (k, v) for (k, v)
                              in attribute.items()]
                ret += "  %s [%s];\n" % (id(node.node), ",".join(attributes))
            elif isinstance(node.node, function.Function):
                attribute = { 'label' : self._function_name(node),
                              'shape': 'record',
                              'style' : 'filled',
                              'fillcolor' : 'aquamarine' }
                attributes = ["%s=\"%s\"" % (k, v) for (k, v)
                              in attribute.items()]
                ret += "  %s [%s];\n" % (id(node.node), ",".join(attributes))
            else:
                attribute = { 'label' : self._function_name(node),
                              'shape': 'record',
                              'style' : 'filled',
                              'fillcolor' : 'cyan' }
                attributes = ["%s=\"%s\"" % (k, v) for (k, v)
                              in attribute.items()]
                ret += "  %s [%s];\n" % (id(node.node), ",".join(attributes))
        for from_node in self.nodes:
            if from_node.out_edges is None: continue
            for to_node in from_node.out_edges:
                if isinstance(from_node.node, variable.Variable):
                    ret += '  %s -> %s:%s;\n' % (id(from_node.node), id(to_node.node), self._variable_elem_name(from_node))
                else:
                    ret += '  %s:%s -> %s;\n' % (id(from_node.node), self._variable_elem_name(to_node), id(to_node.node))


        ret += '  { rank = same; \n'
        for node in self.input_variables:
            attribute = { 'xlabel' : self._variable_name(node),
                          'shape' : 'point' }
            attributes = ["%s=\"%s\"" % (k, v) for (k, v)
                          in attribute.items()]
            ret += "    %s;\n" % (id(node.node))
        ret += '  }\n'
        ret += '  { rank = same; \n'
        for node in self.output_variables:
            attribute = { 'xlabel' : self._variable_name(node),
                          'shape' : 'point' }
            attributes = ["%s=\"%s\"" % (k, v) for (k, v)
                          in attribute.items()]
            ret += "    %s;\n" % (id(node.node))
        ret += '  }\n'
        ret += "}"
        return ret

    def to_mlir(self):
        def _node(node):
            return { b'name': self._variable_elem_name(node),
                     b'dtype': node.node.dtype.str, # TODO: エンディアンとかが入り込むのでfloatとかでいい
                     b'shape': node.node.shape }
        def _function(node):
            aps = self.additional_params[node.node.label] if node.node.label in self.additional_params else None
            params = {}
            if aps is not None:
                def find_params(n, ps):
                    for p in ps:
                        if isinstance(p, str):
                            if isinstance(n.__dict__[p], variable.Variable):
                                yield (str(n.__dict__[p]), encode_ndarray(n.__dict__[p].data))
                            else:
                                yield (p, n.__dict__[p])
                        else:
                            for k,vs in p.items():
                                for (name, param) in find_params(n.__dict__[k], vs):
                                    yield ("%s.%s" % (k, name), param)
                for name, param in find_params(node.node, aps):
                    params[name] = param
            return { b'name': node.node.label,
                     b'inputs': list(map(self._variable_elem_name, reversed(node.in_edges))),
                     b'outputs': list(map(self._variable_elem_name, node.out_edges)),
                     b'params': params }
        sorted_nodes = sorted(self.nodes, key=lambda n: n.no)
        inputs = map(_node, filter(lambda node: isinstance(node.node, variable.Variable), sorted_nodes))
        nodes = map(_node, filter(lambda node: isinstance(node.node, variable.Variable), sorted_nodes))
        edges = map(_function, filter(lambda node: not isinstance(node.node, variable.Variable), sorted_nodes))
        mlir = { b'mlir':
                 { b'version' : 0,
                   b'model' :
                   { b'name': self.model.__class__.__name__,
                     b'generator':
                     { b'name' : 'chainer',
                       b'version': chainer.__version__ },
                     b'inputs': list(map(self._variable_elem_name, self.input_variables)),
                     b'outputs': list(map(self._variable_elem_name, self.output_variables)),
                     b'nodes': list(nodes),
                     b'edges': list(edges) } } }
        return msgpack.packb(mlir)
