from typing import Tuple, List, Optional

import onnx
import numpy as np

from onnx import ModelProto, NodeProto

from ..onnx_utils import get_next_op_node_idx, get_by_name
from .common_fix import replace_blocks, InitsMap


split_op = "Split"
reshape_op = "Reshape"


def get_reshape_shape(model: ModelProto, reshape_node: NodeProto) -> Tuple[int]:
    reshape_size_input_name = reshape_node.input[1]
    leaf_dict = get_by_name(model, [reshape_size_input_name], 'initializer')
    return leaf_dict[reshape_size_input_name].int64_data


def create_half_split_matrices(k: int) -> Tuple[InitsMap, List[str]]:
    k_2 = k//2

    eye = np.eye(k_2, dtype="float32")
    zero = np.zeros((k_2, k_2), dtype="float32")

    up_name = "__split_mat_up_{}".format(k)
    down_name = "__split_mat_down_{}".format(k)
    matrices = {
        up_name: (
            onnx.helper.make_tensor(up_name, onnx.TensorProto.FLOAT, [k, k_2], np.concatenate([eye, zero]).flatten()),
            onnx.helper.make_tensor_value_info(up_name, onnx.TensorProto.FLOAT, [k, k_2]),
        ),
        down_name: (
            onnx.helper.make_tensor(down_name, onnx.TensorProto.FLOAT, [k, k_2], np.concatenate([zero, eye]).flatten()),
            onnx.helper.make_tensor_value_info(down_name, onnx.TensorProto.FLOAT, [k, k_2]),
        )
    }

    return matrices, [up_name, down_name]


def replace_split(model: ModelProto,
                  split_idx: int,
                  subname: str = ""
                 ) -> Tuple[InitsMap, List[NodeProto]]:
    split_node = model.graph.node[split_idx]
    if len(split_node.input) > 1:
        raise ValueError("Split with split arg is not supported")
    if len(split_node.attribute) == 0:
        split_axis = 0
    else:
        split_axis = split_node.attribute[0].i

    inferred_model = onnx.shape_inference.infer_shapes(model)
    value_infos = get_by_name(inferred_model, split_node.input, "value_info")
    shape = tuple(dim.dim_value for dim in value_infos[split_node.input[0]].type.tensor_type.shape.dim)

    if shape[split_axis] % 2 != 0:
        raise ValueError("Cannot reshape on odd size reshaped axis,"
                         " shape {}, axis {}".format(shape, split_axis))

    matrices, matrices_names = create_half_split_matrices(shape[split_axis])

    name_formatter = subname + "_pseudo_split_{}"
    trans_out = name_formatter.format('trans1_out')
    matmul_up_out = name_formatter.format('matmul_up_out')
    matmul_down_out = name_formatter.format('matmul_down_out')

    transpose_perm_0 = list(range(len(shape)))
    transpose_perm_0.append(transpose_perm_0.pop(split_axis))

    transpose_perm_1 = list(range(len(shape)))
    transpose_perm_1.insert(split_axis, transpose_perm_1.pop(-1))

    transpose_node = onnx.helper.make_node(
        "Transpose",
        name=name_formatter.format("transpose0"),
        inputs=[split_node.input[0]],
        outputs=[trans_out],
        perm=transpose_perm_0
    )

    matmul_up_node = onnx.helper.make_node(
        "MatMul",
        name=name_formatter.format("matmul_up"),
        inputs=[trans_out, matrices_names[0]],
        outputs=[matmul_up_out]
    )
    matmul_down_node = onnx.helper.make_node(
        "MatMul",
        name=name_formatter.format("matmul_down"),
        inputs=[trans_out, matrices_names[1]],
        outputs=[matmul_down_out]
    )
    transpose_up_node = onnx.helper.make_node(
        "Transpose",
        name=name_formatter.format("transpose_up"),
        inputs=[matmul_up_out],
        outputs=[split_node.output[0]],
        perm=transpose_perm_1
    )
    transpose_down_node = onnx.helper.make_node(
        "Transpose",
        name=name_formatter.format("transpose_down"),
        inputs=[matmul_down_out],
        outputs=[split_node.output[1]],
        perm=transpose_perm_1
    )

    nodes = [
        transpose_node,
        matmul_up_node,
        matmul_down_node,
        transpose_up_node,
        transpose_down_node
    ]

    return matrices, nodes


def fix_split(model: ModelProto, max_passes: Optional[int] = None):
    """Replace split nodes with equivalent sub-graph.
    Split op is not supported in nnoir, so this function
    reproduces its behavior using Transpose and MatMul.

    Only split in half (no split arg input) is currently supported
    """
    replace_blocks(model,
                   lambda mdl: get_next_op_node_idx(mdl, split_op),
                   replace_split,
                   max_passes)
