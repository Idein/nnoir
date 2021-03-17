from typing import Tuple, Set, Optional, List, Sequence

import onnx
import numpy as np
from onnx import TensorProto, ModelProto, NodeProto

from ..onnx_utils import get_next_op_node_idx, get_nodes_with_var_input, del_nodes, remove_by_name, get_by_name
from .common_fix import InitsMap

QL_op = "QuantizeLinear"
DL_op = "DequantizeLinear"


def get_dl_idx(model: ModelProto, ql_idx: int) -> int:
    res = get_nodes_with_var_input(model, model.graph.node[ql_idx].output[0])
    if len(res) == 0:
        raise ValueError("QL has no corresponding DL node")
    dl_idx, _ = res[0]
    if model.graph.node[dl_idx].op_type != DL_op:
        raise ValueError("QL output is not DL,\nnode name: {}, op_type: {}".format(
            model.graph.node[dl_idx].name, model.graph.node[dl_idx].op_type))
    if model.graph.node[ql_idx].input[1:] != model.graph.node[dl_idx].input[1:]:
        raise ValueError('QL and DL arguments mismatch, op cannot be replaced')
    return dl_idx


def del_nodes_add_unused_inits(model: ModelProto, idx_iterable: Sequence[int], unused_set: Set[str]):
    unused_set.add(model.graph.node[idx_iterable[0]].input[1])
    unused_set.add(model.graph.node[idx_iterable[0]].input[2])
    del_nodes(model, idx_iterable)


def get_Qnode_scale(model: ModelProto, node_idx: int) -> float:
    initializer_name = model.graph.node[node_idx].input[1]
    leaf_dict = get_by_name(model, [initializer_name], 'initializer')
    return leaf_dict[initializer_name].float_data[0]


def get_Qnode_zero(model: ModelProto, node_idx: int) -> int:
    initializer_name = model.graph.node[node_idx].input[2]
    leaf_dict = get_by_name(model, [initializer_name], 'initializer')
    return leaf_dict[initializer_name].int32_data[0]


def create_scalar(name: str, value: float) -> TensorProto:
    return onnx.helper.make_tensor(name, onnx.TensorProto.FLOAT, [], np.array([value], dtype="float32"))
    # return (
    #     onnx.helper.make_tensor(name, onnx.TensorProto.FLOAT, [], np.array([value], dtype="float32")),
    #     onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [])
    # )


def create_pseudoqldl_block(model: ModelProto,
                            ql_idx: int,
                            dl_idx: int,
                            subname: str = ""
                            ) -> Tuple[InitsMap, List[NodeProto]]:
    """version not allowing min different than 0 in Clip op
    """

    scale = get_Qnode_scale(model, ql_idx)
    zero = get_Qnode_zero(model, ql_idx)

    name_formatter = subname + "_pseudoqldl_clip_{}"
    add1 = name_formatter.format('add1_bias')
    add1_out = name_formatter.format('add1_out')
    name_min = name_formatter.format("clip_min")
    name_max = name_formatter.format("clip_max")
    clip_out = name_formatter.format('clip_out')
    add2 = name_formatter.format('add2_bias')

    tensors = {
        add1: create_scalar(add1, zero * scale),
        name_min: create_scalar(name_min, 0.),
        name_max: create_scalar(name_max, 255.*scale),
        add2: create_scalar(add2, - zero*scale)
    }

    add1_node = onnx.helper.make_node(
        "Add",
        name="__pseudoqldl_add1_{}".format(subname),
        inputs=[model.graph.node[ql_idx].input[0], add1],
        outputs=[add1_out]
    )

    clip_node = onnx.helper.make_node(
        "Clip",
        name="__pseudoqldl_clip_{}".format(subname),
        inputs=[add1_out, name_min, name_max],
        outputs=[clip_out]
    )
    add2_node = onnx.helper.make_node(
        "Add",
        name="__pseudoqldl_add2_{}".format(subname),
        inputs=[clip_out, add2],
        outputs=[model.graph.node[dl_idx].output[0]]
    )

    nodes = [
        add1_node,
        clip_node,
        add2_node
    ]

    return tensors, nodes


def fix_quantize(model: ModelProto, max_passes: Optional[int] = None):
    """Replace the quantization blocks in the graph and replace it with
    nnoir-compatible similar blocks.

    The target blocks are QuantizeLinear -> DequantizeLinear blocks,
    with shared `scale` and `zero` parameters.
    (e.g. blocks present in onnx models generated from Cloud AutoML)

    Important Note:
    As at this time rounding operations are not supported in nnoir representation,
    this "fix" does the approximation: `x ~= scale * round(x / scale)`
    so the computation is *not* equivalent.

    Experiments show minor precision decrease on a classification task.
    """
    passes = 0
    next_ql = get_next_op_node_idx(model, QL_op)
    unused_inits = set()  # type: Set[str]
    while next_ql != -1:
        if max_passes is not None and passes >= max_passes:
            break
        passes += 1
        next_dl = get_dl_idx(model, next_ql)
        tensors, nodes = create_pseudoqldl_block(model, next_ql, next_dl, str(passes))
        del_nodes_add_unused_inits(model, (next_dl, next_ql), unused_inits)
        for node in reversed(nodes):
            model.graph.node.insert(next_ql, node)

        model.graph.initializer.extend(tensors.values())

        next_ql = get_next_op_node_idx(model, QL_op)
    remove_by_name(model, unused_inits, 'initializer')
    remove_by_name(model, unused_inits, 'input')
