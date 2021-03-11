from onnx import ModelProto


def _list_dimension_variables(model: ModelProto):
    s = set()
    for x in model.graph.input:
        if x.type.HasField('tensor_type'):
            for dim in x.type.tensor_type.shape.dim:
                if dim.HasField('dim_param'):
                    s.add(dim.dim_param)
    return s


def freeze_dimension_variables(model: ModelProto):
    fix_dimension = {el: 1 for el in _list_dimension_variables(model)}

    for x in model.graph.input:
        if x.type.HasField('tensor_type'):
            for dim in x.type.tensor_type.shape.dim:
                if dim.HasField('dim_param'):
                    v = dim.dim_param
                    dim.ClearField('dim_param')
                    dim.dim_value = fix_dimension[v]


def fix_freeze(model: ModelProto):
    """Freeze non-static dimensions in input to 1.
    (e.g. batch-size)

    Note:
    May produce unexpected results if image size dims were dynamic
    """
    freeze_dimension_variables(model)
