import copy
import tempfile
from itertools import chain
import re
import numpy as np
import onnx
from onnx.shape_inference import infer_shapes
import onnxruntime
from nnoir import *
from nnoir_onnx.operators import *
from .operators.utils import UnsupportedONNXOperation, InvalidONNXData


def tensor_to_narray(tensor):
    arr = []
    storage = onnx.mapping.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE[tensor.data_type]
    storage = onnx.mapping.STORAGE_TENSOR_TYPE_TO_FIELD[storage]
    arr = getattr(tensor, storage)
    if arr == []:
        result = np.frombuffer(tensor.raw_data, dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor.data_type])
    else:
        result = np.array(arr, dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor.data_type])
    shape = tensor.dims if tensor.dims != [] else [1]
    return result.reshape(*shape)


def narray_to_value_info(name, arr):
    return onnx.helper.make_tensor_value_info(name, onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype], arr.shape)


def value_info_to_zero_narray(vi):
    return np.zeros(
        list(map(lambda x: x.dim_value, vi.type.tensor_type.shape.dim)),
        dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[vi.type.tensor_type.elem_type]
    )


def op_for_node(node):
    op_name = 'Op{}'.format(node.op_type)
    if op_name in globals():
        return globals()[op_name](node)
    else:
        raise UnsupportedONNXOperation(node, 'converting from {} is undefined'.format(node.op_type))


class ONNX:

    def __init__(self, path):
        self.model = infer_shapes(onnx.load(path))
        onnx.checker.check_model(self.model)
        # All names MUST adhere to C identifier syntax rules.
        if not re.match(r'^[_A-Za-z][_0-9A-Za-z]*$', self.model.graph.name):
            raise InvalidONNXData('''graph name "{}" is not C identifier.
see https://github.com/onnx/onnx/blob/master/docs/IR.md#names-within-a-graph'''.format(self.model.graph.name))
        self._rename_to_c_ident()
        self._check_opset_compatibility()
        self.sess = onnxruntime.InferenceSession(path)
        constant_nodes = self._list_constant_nodes()
        self.nodes = self._try_run()
        variables = self._statically_unknown_variables()
        if variables != []:
            raise UnsupportedONNXOperation(
                variables, "This ONNX model includes dimension variables. Try to remove them by assignment by `freeze_onnx`")
        self.constant_nodes = self._eval_nodes(constant_nodes)

    def _rename_to_c_ident(self):
        m = infer_shapes(copy.deepcopy(self.model))
        value_names = [i.name for i in m.graph.input] + [v.name for v in m.graph.value_info]
        # Initializer id is not restricted C identifier syntax rules.
        for initializer in self.model.graph.initializer:
            rename_step = 0
            rename_prefix = 'v_from_initializer'
            rename_content = initializer.name
            if re.match(r'^[_A-Za-z][_0-9A-Za-z]*$', rename_content):
                rename_prefix += '_plain'
            else:
                rename_content = ''.join(map(lambda c: 'x{:02x}'.format(ord(c)), rename_content))
                rename_prefix += '_encoded'
            rename_candidate = "{}_{}_{}".format(rename_prefix, rename_step, rename_content)
            while True:
                if rename_candidate not in value_names:
                    value_names.append(rename_candidate)
                    break
                rename_step += 1
                rename_candidate = "{}_{}_{}".format(rename_prefix, rename_step, rename_content)
            # rename initializer.name -> rename_candidate
            for n in self.model.graph.node:
                for i in range(len(n.input)):
                    if n.input[i] == initializer.name:
                        n.input[i] = rename_candidate
            initializer.name = rename_candidate

    def _check_opset_compatibility(self):
        ops = set()
        for op in self.model.graph.node:
            ops.add(str(op.op_type))

        opset_version = self.model.opset_import[0].version

        # Resize is compatible only with opset >= 11
        if 'Resize' in ops and opset_version < 11:
            raise InvalidONNXData('Resize operator from opset version < 11 is not supported')

    def _try_run(self):
        m = infer_shapes(copy.deepcopy(self.model))
        values = [v.name for v in m.graph.value_info]
        m.graph.output.extend(m.graph.input)
        m.graph.output.extend(m.graph.value_info)

        def tensor_to_value_info(t):
            return onnx.helper.make_tensor_value_info(t.name, t.data_type, None)
        m.graph.output.extend(list(map(tensor_to_value_info, m.graph.initializer)))
        inits = [v.name for v in m.graph.initializer]
        input_values = [v for v in m.graph.input if v.name not in inits]
        outputs = [
            *[v.name for v in m.graph.input],
            *[v.name for v in m.graph.value_info],
            *[v.name for v in m.graph.output],
        ]
        with tempfile.NamedTemporaryFile() as f:
            onnx.save(m, f.name)
            sess = onnxruntime.InferenceSession(f.name)
            dummy_inputs = {i.name: value_info_to_zero_narray(i) for i in input_values}
            return {name: value for name, value in zip(outputs, sess.run(outputs, dummy_inputs))}

    def _find(self, p, xs, default=None):
        return next(filter(p, xs), default)

    def _find_initializer(self, name):
        return self._find(lambda n: name == n.name, self.model.graph.initializer)

    def _has_initializer(self, name):
        return self._find_initializer(name) is not None

    def _find_generator(self, name):
        return self._find(lambda n: name in n.output, self.model.graph.node)

    def _find_input(self, name):
        return self._find(lambda n: name == n.name, self.model.graph.input)

    def _has_input(self, name):
        return self._find_input(name) is not None

    def to_NNOIR(self):
        inputs = list(map(lambda x: x.name, self.sess.get_inputs()))
        outputs = list(map(lambda x: x.name, self.sess.get_outputs()))
        functions = self._to_NNOIR_functions()
        nodes = [Value(n, self.nodes[n])
                 for n in set(chain.from_iterable(map(lambda x: x.inputs + x.outputs, functions)))]

        # rename to C ident (some frameworks don't satisfy the onnx spec.)
        renaming_table = dict([(n.name, 'v{}'.format(i).encode('utf-8')) for i, n in enumerate(nodes)])

        def rename(x):
            return renaming_table[x]
        inputs = list(map(rename, inputs))
        outputs = list(map(rename, outputs))

        def rename_function(e):
            e.inputs = list(map(rename, e.inputs))
            e.outputs = list(map(rename, e.outputs))
            return e
        functions = list(map(rename_function, functions))

        def rename_node(n):
            n.name = rename(n.name)
            return n
        nodes = list(map(rename_node, nodes))

        return NNOIR(
            self.model.graph.name.encode('utf-8'),
            self.model.producer_name,
            self.model.producer_version,
            inputs,
            outputs,
            nodes,
            functions
        )

    def _eval_nodes(self, nodes):
        m = copy.deepcopy(self.model)
        for n in m.graph.output:
            m.graph.output.remove(n)
        m.graph.output.extend(map(lambda n: narray_to_value_info(n, self.nodes[n]), nodes))
        with tempfile.NamedTemporaryFile() as f:
            onnx.save(m, f.name)
            dummy_sess = onnxruntime.InferenceSession(f.name)
        inputs = dict([(x.name, self.nodes[x.name]) for x in dummy_sess.get_inputs()])
        output_names = list(map(lambda x: x.name, dummy_sess.get_outputs()))
        if output_names != []:
            result = dummy_sess.run(output_names, inputs)
        else:
            result = []
        return dict(zip(output_names, result))

    def test(self):
        with tempfile.NamedTemporaryFile() as tmpf:
            m = copy.deepcopy(self.model)
            for n in m.graph.output:
                m.graph.output.remove(n)
            m.graph.output.extend(map(lambda nv: narray_to_value_info(nv[0], nv[1]), self.nodes.items()))
            for n in m.graph.input:
                m.graph.output.remove(n)
            onnx.save(m, tmpf.name)
            sess = onnxruntime.InferenceSession(tmpf.name)
            inputs = {x.name: np.zeros(tuple(x.shape), dtype=np.float32) for x in sess.get_inputs()}
            outputs = [x.name for x in sess.get_inputs()]
            results = sess.run(outputs, inputs)

    def _to_NNOIR_functions(self):
        outputs = list(map(lambda x: x.name, self.sess.get_outputs()))
        visited = []
        functions = []
        while outputs != []:
            o = outputs.pop(0)
            if o in visited:
                continue
            visited.append(o)
            generator = self._find_generator(o)
            if generator is not None:
                function = op_for_node(generator).to_function(self.nodes, self.constant_nodes)
                inputs = list(chain.from_iterable(map(lambda x: x.inputs, function)))
                outputs += inputs
                functions += function
            initializer = self._find_initializer(o)
            if initializer is not None:
                raise UnsupportedONNXOperation(node, 'converting from Constant is undefined')
        return functions

    def _list_constant_nodes(self):
        outputs = list(map(lambda x: x.name, self.sess.get_outputs()))

        def dfs(visited, nodes, result):
            for n in nodes:
                if self._has_initializer(n):
                    result.append(n)
                elif self._has_input(n):
                    pass
                else:
                    generator = self._find_generator(n)
                    if generator.op_type == 'Shape':  # In nnoir, array shape is known information.
                        result.append(n)
                    else:
                        next_nodes = []
                        if hasattr(generator, 'input'):
                            next_nodes = [i for i in generator.input if i not in visited]
                        dfs(visited, next_nodes, result)
                        if hasattr(generator, 'input'):
                            if all([i in result for i in generator.input]):
                                for o in generator.output:
                                    result.append(o)
                        else:
                            for o in generator.output:
                                result.append(o)
                visited.append(n)
        result = []
        dfs([], outputs, result)
        return result

    def _statically_unknown_variables(self):
        variables = []
        for n in self.nodes:
            _input = self._find_input(n)
            if _input and hasattr(_input.type, 'tensor_type'):
                dims = _input.type.tensor_type.shape.dim
                variables += filter(lambda x: x.dim_param != '', dims)
        return variables


def to_dummy_input(x):
    if hasattr(x.type, 'tensor_type'):
        if x.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
            return np.zeros(tuple(map(lambda d: d.dim_value, x.type.tensor_type.shape.dim)), dtype=np.float32)
        else:
            raise 'unsupported'
    else:
        raise 'unsupported'
