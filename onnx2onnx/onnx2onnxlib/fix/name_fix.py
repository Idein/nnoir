from onnx import ModelProto


def fix_name(model: ModelProto):
    """Fix graph name.
    Hyphens are not authorized in C identifier,
    so this function replaces it with underscore.

    (lots of services produce onnx models with incorrect graph name)
    """
    model.graph.name = model.graph.name.replace('-', '_')  # change name for C identifier
