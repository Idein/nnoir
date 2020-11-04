from typing import Dict, Set

def list_dimension_variables(model) -> Set[str]:
    s = set()
    for x in model.graph.input:
        if x.type.HasField('tensor_type'):
            for dim in x.type.tensor_type.shape.dim:
                if dim.HasField('dim_param'):
                    s.add(dim.dim_param)
    return s

