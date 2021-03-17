from typing import Callable, Optional, List, Tuple, Dict
from onnx import ModelProto, TensorProto, NodeProto, ValueInfoProto

from ..onnx_utils import del_nodes

IdxGetter = Callable[[ModelProto], int]
InitsMap = Dict[str, Tuple[TensorProto, ValueInfoProto]]

FixFunc = Callable[[ModelProto, int, str],
                   Tuple[InitsMap, List[NodeProto]]]
"""A function FixFunc has the following signature:
It should take has arguments:
- the model to operate on
- the index of the node to fix
- an optional string used as prefix on names created by the fixing function

It returns a tuple of two elements:
- a mapping of string to couple (tensorproto, valueinfo) of added parameters
        that should be appended respectively to the model initializer and input
- a list of nodes that should be added to the graph nodes

"""


def replace_blocks(model: ModelProto, node_idx_getter: IdxGetter, replace_func: FixFunc, max_passes: Optional[int] = None):
    passes = 0
    next_idx = node_idx_getter(model)
    new_inits = {}  # type: Dict[str, Tuple[TensorProto, ValueInfoProto]]
    while next_idx != -1:
        if max_passes is not None and passes >= max_passes:
            break
        passes += 1

        tensor_dict, nodes = replace_func(model, next_idx, str(passes))

        del_nodes(model, [next_idx])
        for node in reversed(nodes):
            model.graph.node.insert(next_idx, node)
        new_inits.update(tensor_dict)
        next_idx = node_idx_getter(model)
    for tensor, valueinfo in new_inits.values():
        model.graph.initializer.append(tensor)
        model.graph.input.append(valueinfo)
