import argparse
import sys

import onnx
from nnoir_onnx import ONNX, utils


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX to NNOIR Converter")
    parser.add_argument(
        "-o", "--output", dest="output", type=str, required=True, metavar="NNOIR", help="output(NNOIR) file path"
    )
    parser.add_argument(
        "--graph_name",
        dest="graph_name",
        type=str,
        required=False,
        metavar="C_IDENT",
        help="new graph name (it must be a C identifer.)",
    )
    parser.add_argument(
        "--fix_dimension",
        type=utils.parse_assign,
        dest="fix_dimension",
        required=False,
        help="assign dimension variables (like W=2,H=4. Cannot include spaces)",
    )
    parser.add_argument(dest="input", type=str, metavar="ONNX", help="input(ONNX) file path")
    args = parser.parse_args()
    try:
        ONNX(args.input, args.graph_name, args.fix_dimension).to_NNOIR().dump(args.output)
    except Exception as e:
        sys.exit(f"Error: {e}")
