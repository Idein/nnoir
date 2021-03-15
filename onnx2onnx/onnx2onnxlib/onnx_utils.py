from typing import List, Tuple, Iterable, Dict, Union

from onnx import ModelProto, NodeProto, ValueInfoProto, TensorProto
from .utils import index_of

Leaf = Union[ValueInfoProto, TensorProto]


def get_next_op_node_idx(model: ModelProto, op: str) -> int:
    for idx, node in enumerate(model.graph.node):
        if node.op_type == op:
            return idx
    return -1


def _get_nodes_with_var(model: ModelProto,
                        var_name: str,
                        attribute_name: str) -> List[Tuple[int, NodeProto]]:
    res = []
    for idx, node in enumerate(model.graph.node):
        if var_name in getattr(node, attribute_name):
            res.append((idx, index_of(getattr(node, attribute_name), var_name)))
    return res


def get_nodes_with_var_input(model: ModelProto, var_name: str) -> List[Tuple[int, NodeProto]]:
    return _get_nodes_with_var(model, var_name, 'input')


def get_nodes_with_var_output(model: ModelProto, var_name: str) -> List[Tuple[int, NodeProto]]:
    return _get_nodes_with_var(model, var_name, 'output')


def remove_by_name(model: ModelProto, name_iterable: Iterable[str], graph_attr: str):
    indices = set()
    for idx, el in enumerate(getattr(model.graph, graph_attr)):
        if el.name in name_iterable:
            indices.add(idx)

    for idx in sorted(indices, reverse=True):
        getattr(model.graph, graph_attr).pop(idx)


def get_by_name(model: ModelProto, name_iterable: Iterable[str], graph_attr: str) -> Dict[str, Leaf]:
    ret = {}
    for el in getattr(model.graph, graph_attr):
        if el.name in name_iterable:
            ret[el.name] = el
    return ret


def del_nodes(model: ModelProto, idx_iterable: Iterable[int]):
    for node in [model.graph.node[node_idx] for node_idx in idx_iterable]:
        model.graph.node.remove(node)
