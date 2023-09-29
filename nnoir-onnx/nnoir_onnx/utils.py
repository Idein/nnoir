from typing import Dict, Set

import onnx

from .operators.utils import UnknownSizedVariable


def list_dimension_variables(model: onnx.ModelProto) -> Set[str]:
    s = set()
    for x in model.graph.input:
        if x.type.HasField("tensor_type"):
            for dim in x.type.tensor_type.shape.dim:
                if dim.HasField("dim_param"):
                    s.add(dim.dim_param)
    return s


def freeze_dimension_variables(model: onnx.ModelProto, fix_dimension: Dict[str, int]) -> onnx.ModelProto:
    s = list_dimension_variables(model)
    diff = s.difference(set(fix_dimension.keys()))
    if len(diff) != 0:
        raise UnknownSizedVariable("missing variables: " + str(diff))

    for x in model.graph.input:
        if x.type.HasField("tensor_type"):
            for dim in x.type.tensor_type.shape.dim:
                if dim.HasField("dim_param"):
                    v = dim.dim_param
                    dim.ClearField("dim_param")
                    dim.dim_value = fix_dimension[v]

    return model


def parse_assign(s: str) -> Dict[str, int]:
    d: Dict[str, int] = dict()
    item: str
    for item in s.split(","):
        [v, n] = item.split("=")
        d[v] = int(n)
    return d
