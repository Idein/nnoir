import copy
import re
import tempfile
from itertools import chain

import numpy as np
import onnx
import onnxruntime
from nnoir import *
from nnoir_onnx.operators import *

from .operators.utils import InvalidONNXData, UnknownSizedVariable, UnsupportedONNXOperation
from .utils import freeze_dimension_variables, list_dimension_variables


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
        dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[vi.type.tensor_type.elem_type],
    )


class ONNX:
    def __init__(self, path, graph_name=None, fix_dimension=None):
        self.onnx_path = path
        self.model = onnx.load(path)
        if graph_name is not None:
            self.model.graph.name = graph_name
        if fix_dimension is not None:
            self.model = freeze_dimension_variables(self.model, fix_dimension)
        c_keywords = [
            "auto",
            "break",
            "case",
            "char",
            "const",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extern",
            "float",
            "for",
            "goto",
            "if",
            "inline",
            "int",
            "long",
            "register",
            "restrict",
            "return",
            "short",
            "signed",
            "sizeof",
            "static",
            "struct",
            "switch",
            "typedef",
            "union",
            "unsigned",
            "void",
            "volatile",
            "while",
            "_Alignas",
            "_Alignof",
            "_Atomic",
            "_Bool",
            "_Complex",
            "_Generic",
            "_Imaginary",
            "_Noreturn",
            "_Static_assert",
            "_Thread_local",
        ]
        onnx.checker.check_model(self.model)
        # All names MUST adhere to C identifier syntax rules.
        if not re.match(r"^[_A-Za-z][_0-9A-Za-z]*$", self.model.graph.name) or self.model.graph.name in c_keywords:
            raise InvalidONNXData(
                f"""graph name "{self.model.graph.name}" is not C identifier.
see https://github.com/onnx/onnx/blob/master/docs/IR.md#names-within-a-graph.
You can override the graph name with the `--graph_name` option."""
            )
        if self.model.graph.name == "main":
            print(
                """Warning: the graph name "main" conflicts with the main function of C, if you use the nnoir from C.
You can override the graph name with the `--graph_name` option."""
            )
        variables = list_dimension_variables(self.model)
        if len(variables) != 0:
            raise UnknownSizedVariable(
                f"""This ONNX model includes dimension variables.
{variables}
Set the values with the `--fix_dimension` option."""
            )
        self._rename_to_c_ident()
        sess_opts = onnxruntime.SessionOptions()
        sess_opts.log_severity_level = 3
        self.sess = onnxruntime.InferenceSession(path, sess_options=sess_opts)
        constant_nodes = self._list_constant_nodes()
        self.nodes = self._try_run(constant_nodes)
        self.constant_nodes = {n: self.nodes[n] for n in constant_nodes}
        self.opset_version = self.model.opset_import[0].version

    def _internal_values_info(self, model):
        values = list(set([v for n in model.graph.node for v in n.output]))
        return [onnx.helper.make_empty_tensor_value_info(v) for v in values]

    def _rename_to_c_ident(self):
        m = copy.deepcopy(self.model)
        value_names = [i.name for i in m.graph.input] + [v.name for v in self._internal_values_info(m)]
        # Initializer id is not restricted C identifier syntax rules.
        for initializer in self.model.graph.initializer:
            rename_step = 0
            rename_prefix = "v_from_initializer"
            rename_content = initializer.name
            if re.match(r"^[_A-Za-z][_0-9A-Za-z]*$", rename_content):
                rename_prefix += "_plain"
            else:
                rename_content = "".join(map(lambda c: f"x{ord(c):02x}", rename_content))
                rename_prefix += "_encoded"
            rename_candidate = f"{rename_prefix}_{rename_step}_{rename_content}"
            while True:
                if rename_candidate not in value_names:
                    value_names.append(rename_candidate)
                    break
                rename_step += 1
                rename_candidate = f"{rename_prefix}_{rename_step}_{rename_content}"
            # rename initializer.name -> rename_candidate
            for n in self.model.graph.node:
                for i in range(len(n.input)):
                    if n.input[i] == initializer.name:
                        n.input[i] = rename_candidate
            initializer.name = rename_candidate

    def _try_run(self, constant_nodes):
        model = copy.deepcopy(self.model)
        while len(model.graph.output) > 0:
            model.graph.output.pop(0)
        inits = [v.name for v in model.graph.initializer]
        input_values = [v for v in model.graph.input if v.name not in inits]
        dummy_inputs = {i.name: value_info_to_zero_narray(i) for i in input_values}
        outputs = [
            *[v for v in model.graph.input],
            *self._internal_values_info(model),
        ]

        result = copy.deepcopy(dummy_inputs)
        for t in model.graph.initializer:
            result[t.name] = tensor_to_narray(t)
        with tempfile.NamedTemporaryFile() as f:

            while len(model.graph.node) > 0:

                # create single operation graph
                m = copy.deepcopy(model)
                while len(m.graph.node) > 0:
                    m.graph.node.pop(0)
                while len(m.graph.input) > 0:
                    m.graph.input.pop(0)
                while len(m.graph.initializer) > 0:
                    m.graph.initializer.pop(0)

                node = model.graph.node.pop(0)
                m.graph.node.append(node)

                inputs = [i for i in node.input if i not in inits and len(i) > 0]
                m.graph.input.extend([n for n in model.graph.input if n.name in inputs])

                outputs = node.output
                m.graph.output.extend([onnx.helper.make_empty_tensor_value_info(v) for v in outputs])

                initializers = [i for i in node.input if i in inits]
                m.graph.initializer.extend([n for n in model.graph.initializer if n.name in initializers])

                onnx.save(m, f.name)
                sess_opts = onnxruntime.SessionOptions()
                sess_opts.log_severity_level = 3
                sess = onnxruntime.InferenceSession(f.name, sess_options=sess_opts)
                for k, v in zip(outputs, sess.run(outputs, {i: dummy_inputs[i] for i in inputs})):
                    if k not in constant_nodes:
                        # save memory usage
                        v = np.broadcast_to(np.zeros(1, dtype=v.dtype), (1,) if v.ndim == 0 else v.shape)
                    result[k] = v
                    dummy_inputs[k] = v
                    model.graph.input.append(narray_to_value_info(k, v))

            return result

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
        try:
            functions = self._to_NNOIR_functions()
        except UnsupportedONNXOperation as e:
            self._dump_dot()
            raise e
        nodes = [Value(n, self.nodes[n]) for n in set(chain.from_iterable(map(lambda x: x.inputs + x.outputs, functions)))]

        # rename to C ident (some frameworks don't satisfy the onnx spec.)
        renaming_table = {n.name: f"v{i}".encode("utf-8") for i, n in enumerate(nodes)}

        def rename(x):
            try:
                return renaming_table[x]
            except Exception as e:
                raise RuntimeError(f"not found key {x} in renaming_table")

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
            self.model.graph.name.encode("utf-8"),
            self.model.producer_name,
            self.model.producer_version,
            inputs,
            outputs,
            nodes,
            functions,
        )

    def _eval_nodes(self, nodes):
        m = copy.deepcopy(self.model)
        for n in m.graph.output:
            m.graph.output.remove(n)
        m.graph.output.extend(map(lambda n: narray_to_value_info(n, self.nodes[n]), nodes))
        with tempfile.NamedTemporaryFile() as f:
            onnx.save(m, f.name)
            sess_opts = onnxruntime.SessionOptions()
            sess_opts.log_severity_level = 3
            dummy_sess = onnxruntime.InferenceSession(f.name, sess_options=sess_opts)
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
            sess_opts = onnxruntime.SessionOptions()
            sess_opts.log_severity_level = 3
            sess = onnxruntime.InferenceSession(tmpf.name, sess_options=sess_opts)
            inputs = {x.name: np.zeros(tuple(x.shape), dtype=np.float32) for x in sess.get_inputs()}
            outputs = [x.name for x in sess.get_inputs()]
            results = sess.run(outputs, inputs)

    def op_for_node(self, node):
        op_name = f"Op{node.op_type}"
        if op_name in globals():
            return globals()[op_name](node, self.opset_version)
        else:
            raise UnsupportedONNXOperation(node, f"converting from {node.op_type} is undefined")

    def _to_NNOIR_functions(self):
        outputs = list(map(lambda x: x.name, self.sess.get_outputs()))
        visited = []
        known_generator = []
        functions = []
        while outputs != []:
            o = outputs.pop(0)
            if o in visited:
                continue
            visited.append(o)
            generator = self._find_generator(o)
            if generator in known_generator:
                continue
            if generator is not None:
                function = self.op_for_node(generator).to_function(self.nodes, self.constant_nodes)

                inputs = list(chain.from_iterable(map(lambda x: x.inputs, function)))
                outputs += inputs
                functions += function
                known_generator.append(generator)
            initializer = self._find_initializer(o)
            if initializer is not None:
                raise UnsupportedONNXOperation(initializer, "converting from Constant is undefined")
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
                    if generator.op_type == "Shape":  # In nnoir, array shape is known information.
                        result.append(n)
                    else:
                        next_nodes = []
                        if hasattr(generator, "input"):
                            next_nodes = [i for i in generator.input if i not in visited and len(i) > 0]
                        dfs(visited, next_nodes, result)
                        if hasattr(generator, "input"):
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

    def _dot_box_color(self, node):
        if not all([o in self.constant_nodes for o in node.output]):
            try:
                _ = self.op_for_node(node).to_function(self.nodes, self.constant_nodes)
            except Exception:
                return "sandybrown"

            return "aquamarine"

        else:
            return "white"

    def _dump_dot(self):
        dot_path = f"{self.onnx_path}.dot"
        ln = "&#92;l"

        value_name_table = NameTable("val")
        function_name_table = NameTable("fun")

        def is_used(name):
            for n in self.model.graph.node:
                if name in n.input:
                    return True
            return False

        model_input_names = [v.name for v in self.model.graph.input if is_used(v.name)]
        model_output_names = [v.name for v in self.model.graph.output]

        model_input = "\n  ".join([f'{value_name_table[n]} [label="{n}",shape="oval"];' for n in model_input_names])
        model_output = "\n  ".join([f'{value_name_table[n]} [label="{n}",shape="oval"];' for n in model_output_names])
        op_dot = []
        for n in self.model.graph.node:
            op_output_values = "\n  ".join(
                [f'{value_name_table[o]} [xlabel="{o}",shape="point"];' for o in n.output if o not in model_output_names]
            )
            op_input = ", ".join([i for i in n.input if not self._has_initializer(i)])
            op_output = ", ".join([o for o in n.output])
            op_ident = function_name_table[f"{n.name} {n.op_type} {op_input} {op_output}"]
            op_label = f"{n.op_type}{ln}name: {n.name}{ln}input: {op_input}{ln}output: {op_output}{ln}"
            op_info = (
                f'{op_ident} [label="{{{op_label}}}", shape="record", style="filled", fillcolor="{self._dot_box_color(n)}"];'
            )
            op_input_edge = "  ".join(
                [f"{value_name_table[i]} -> {op_ident};" for i in n.input if not self._has_initializer(i)]
            )
            op_output_edge = "  ".join([f"{op_ident} -> {value_name_table[o]};" for o in n.output])
            op_dot.append(
                f"""{op_output_values}
  {op_info}
  {op_input_edge}
  {op_output_edge}"""
            )

        operators = "\n  ".join(op_dot)

        dot = f"""digraph graphname {{ rankdir=TB;
  subgraph input {{
    {model_input}
  }}
  subgraph output {{
    {model_output}
  }}
  {operators}
}}
"""
        with open(dot_path, "w") as f:
            f.write(dot)
        print(
            f"""############################################################################################
  Generate {dot_path}.
  Check unsupported operators by `dot -O -Tsvg {dot_path}`.
  The color of the node means
    - green  -> supported operator
    - orange -> unsupported operator
    - white  -> operator which is folded into a constant value.

  Extract the subgraph which has only supported operators. (Use a tool such as onnigiri.)
  Then convert the subgraph to nnoir.
############################################################################################"""
        )


def to_dummy_input(x):
    if hasattr(x.type, "tensor_type"):
        if x.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
            return np.zeros(
                tuple(map(lambda d: d.dim_value, x.type.tensor_type.shape.dim)),
                dtype=np.float32,
            )
        else:
            raise "unsupported"
    else:
        raise "unsupported"


class NameTable:
    def __init__(self, prefix: str) -> None:
        self.tbl = dict()
        self.prefix = prefix

    def __getitem__(self, key: str) -> str:
        if not (key in self.tbl):
            self.tbl[key] = f"{self.prefix}{len(self.tbl)}"
        return self.tbl[key]
