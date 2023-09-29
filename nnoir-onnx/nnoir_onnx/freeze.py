import argparse
from typing import Dict, Set

import onnx
from nnoir_onnx import utils


def command_list(args: argparse.Namespace) -> None:
    model = onnx.load(args.input)
    s = utils.list_dimension_variables(model)
    if len(s) != 0:
        print(s)


def command_freeze(args: argparse.Namespace) -> None:
    model = onnx.load(args.input)
    fixed_model = utils.freeze_dimension_variables(model, args.fix_dimension)
    onnx.save(fixed_model, args.output)


def freeze() -> None:
    print("Warning: freeze_onnx is deprecated. Instead use `onnx2nnoir --fix_dimension`.")
    parser = argparse.ArgumentParser(description="ONNX Freezer")
    subparsers = parser.add_subparsers()

    # freeze
    parser_freeze = subparsers.add_parser("freeze", help="create freezed (statically sized) onnx")
    parser_freeze.add_argument(
        "-o", "--output", dest="output", type=str, required=True, metavar="ONNX", help="output(ONNX) file path"
    )
    parser_freeze.add_argument(
        "--fix-dimension",
        type=utils.parse_assign,
        dest="fix_dimension",
        required=True,
        help="assign statically unknown variables (like W=2,H=4. Cannot include spaces)",
    )
    parser_freeze.add_argument(dest="input", type=str, metavar="ONNX", help="input(ONNX) file path")
    parser_freeze.set_defaults(handler=command_freeze)

    # list
    parser_list = subparsers.add_parser("list", help="list all statically unknown sized variables")
    parser_list.add_argument(dest="input", type=str, metavar="ONNX", help="input(ONNX) file path")

    parser_list.set_defaults(handler=command_list)

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()
