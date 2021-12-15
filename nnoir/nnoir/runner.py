import argparse
import sys

import numpy as np
from nnoir import NNTC, TestValue, load


def pack_values(ns, vs):
    return [TestValue(n, v) for (n, v) in zip(ns, vs)]


def generate_inputs(vs, seed, base, scale):
    inputs = []
    np.random.seed(seed)
    for v in vs:
        x = base + scale * np.random.randn(*v.shape).astype(np.float32)
        inputs.append(x)
    return inputs


def main():
    parser = argparse.ArgumentParser(description="NNTC generator")
    parser.add_argument(dest="nnoir", type=str, metavar="NNOIR", help="input nnoir file")
    parser.add_argument("-o", "--output", dest="output", type=str, required=True, metavar="PATH", help="output file path")
    parser.add_argument("--seed", dest="seed", type=int, metavar="INT", help="seed for inputs generation", default=0)
    parser.add_argument("--base", dest="base", type=float, metavar="FLOAT", help="modify randn", default=0.0)
    parser.add_argument("--scale", dest="scale", type=float, metavar="FLOAT", help="modify randn", default=1.0)

    args = parser.parse_args()

    nnoir_model = load(args.nnoir)
    if nnoir_model.name == b"main":
        sys.exit(f"Error: invalid model name ({nnoir_model.name})")

    input_values = [[v for v in nnoir_model.values if v.name == n][0] for n in nnoir_model.inputs]

    inputs = generate_inputs(input_values, args.seed, args.base, args.scale)
    outputs = nnoir_model.run(*inputs)

    tc = NNTC(nnoir_model.name, pack_values(nnoir_model.inputs, inputs), pack_values(nnoir_model.outputs, outputs))
    tc.dump(args.output)
