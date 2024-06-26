import argparse
import sys

import onnx
from blackonnx import fix

fix_funcs = [func for func in fix.__dict__ if func.startswith("fix_")]


def main():
    parser = argparse.ArgumentParser("Apply fixes to onnx model")
    parser.add_argument(dest="input", type=str, help="path to input onnx file to correct")
    parser.add_argument(
        "-o", "--output", dest="output", type=str, default="", help="output name (default is input name with '_fixed' postfix)"
    )
    parser.add_argument(
        "--fixes",
        nargs="+",
        default=[],
        help="list with white-spaces of fixes to apply. (Default applies all fixes)"
        "Possible values:\n '{}'. (order matters)".format("', '".join(fix_funcs)),
    )
    options = parser.parse_args()

    if not set(fix_funcs).issuperset(options.fixes):
        raise ValueError(
            "unknown fix: {}\nPossible values are: '{}'".format(
                set(options.fixes).difference(fix_funcs), "', '".join(fix_funcs)
            )
        )

    if len(options.fixes) == 0:
        options.fixes = fix_funcs

    if options.output == "":
        options.output = options.input[:-5] + "_fixed.onnx"
    try:
        model = onnx.load(options.input)
        for fix_name in options.fixes:
            getattr(fix, fix_name)(model)
        onnx.save(model, options.output)
    except Exception as e:
        sys.exit("Error: {}".format(e))
