import argparse
import sys
from typing import List

import numpy as np
from nnoir import NNTC, TestValue, Value, load
from numpy.typing import NDArray


def pack_values(ns: List[bytes], vs: List[NDArray[np.float32]]) -> List[TestValue]:
    return [TestValue(n, v) for (n, v) in zip(ns, vs)]


def generate_inputs(vs: List[Value], seed: int, base: float, scale: float) -> List[NDArray[np.float32]]:
    inputs = []
    np.random.seed(seed)
    for v in vs:
        x = base + scale * np.random.randn(*v.shape).astype(np.float32)  # type: ignore
        inputs.append(x)
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="NNTC generator")
    parser.add_argument(dest="nnoir", type=str, metavar="NNOIR", help="input nnoir file")
    parser.add_argument("-o", "--output", dest="output", type=str, required=True, metavar="PATH", help="output file path")
    parser.add_argument("--seed", dest="seed", type=int, metavar="INT", help="seed for inputs generation", default=0)
    parser.add_argument("--base", dest="base", type=float, metavar="FLOAT", help="modify randn", default=0.0)
    parser.add_argument("--scale", dest="scale", type=float, metavar="FLOAT", help="modify randn", default=1.0)

    args = parser.parse_args()

    nnoir_model = load(args.nnoir)
    if nnoir_model.name == b"main":
        sys.exit(f"Error: invalid model name ({nnoir_model.name.decode()})")

    input_values = [[v for v in nnoir_model.values if v.name == n][0] for n in nnoir_model.inputs]

    inputs = generate_inputs(input_values, args.seed, args.base, args.scale)
    outputs = nnoir_model.run(*inputs)

    tc = NNTC(nnoir_model.name, pack_values(nnoir_model.inputs, inputs), pack_values(nnoir_model.outputs, outputs))
    tc.dump(args.output)
