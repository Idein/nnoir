import argparse
import io
import sys

import msgpack
import numpy


def function_label(function):
    ret = "{{" + (
        "|".join(
            [function[b"name"].decode()]
            + list(map(lambda v: "<a" + str(v[0]) + v[1].decode() + ">", enumerate(function[b"inputs"])))
        )
    )
    if b"W" in function[b"params"]:
        ret += "|W"
    if b"b" in function[b"params"]:
        ret += "|b"
    ret += "}"

    def find_params(params):
        for k, v in params.items():
            if type(v) is dict and b"ndarray" in v:
                array = numpy.load(io.BytesIO(v[b"ndarray"]))
                yield k.decode() + " shape: " + str(array.shape)
            elif type(v) is dict and b"nnoir" in v:
                yield k.decode() + ": " + v[b"nnoir"][b"model"][b"name"].decode("utf-8")
            else:
                yield k.decode() + ": " + str(v)

    params_str = "&#92;l".join(find_params(function[b"params"]))
    if params_str != "":
        ret += "|{%s&#92;l}" % params_str
    ret += "|{%s}}" % "|".join(map(lambda v: "<" + v.decode() + ">", reversed(function[b"outputs"])))
    return ret


def function_name(function):
    inputs = "".join(map(lambda v: v.decode(), reversed(function[b"inputs"])))
    outputs = "".join(map(lambda v: v.decode(), reversed(function[b"outputs"])))
    return "{}_{}_{}".format(function[b"name"].decode(), inputs, outputs)


def value_name(value):
    return "{} {}".format(value[b"name"].decode(), tuple(value[b"shape"]))


def to_dot(nnoir, rankdir="TB"):
    inputs = nnoir[b"nnoir"][b"model"][b"inputs"]
    outputs = nnoir[b"nnoir"][b"model"][b"outputs"]
    values = nnoir[b"nnoir"][b"model"][b"values"]
    functions = nnoir[b"nnoir"][b"model"][b"functions"]

    ret = "digraph graphname { rankdir=%s;\n" % rankdir
    ret += "  subgraph input {\n"
    for i, var in enumerate(inputs):
        attribute = {"xlabel": "input%d = %s" % (i, var.decode()), "shape": "point"}
        attributes = ['%s="%s"' % (k, v) for (k, v) in attribute.items()]
        ret += "    %s [%s];\n" % (var.decode(), ",".join(attributes))
    ret += "  }\n"
    ret += "  subgraph output {\n"
    for i, var in enumerate(outputs):
        attribute = {"xlabel": "output%d = %s" % (i, var.decode()), "shape": "point"}
        attributes = ['%s="%s"' % (k, v) for (k, v) in attribute.items()]
        ret += "    %s [%s];\n" % (var.decode(), ",".join(attributes))
    ret += "  }\n"
    for value in values:
        if value[b"name"] in inputs:
            continue
        if value[b"name"] in outputs:
            continue
        attribute = {"xlabel": value_name(value), "shape": "point"}
        attributes = ['%s="%s"' % (k, v) for (k, v) in attribute.items()]
        ret += "  %s [%s];\n" % (value[b"name"].decode(), ",".join(attributes))
    for function in functions:
        # output function
        attribute = {"label": function_label(function), "shape": "record", "style": "filled", "fillcolor": "aquamarine"}
        attributes = ['%s="%s"' % (k, v) for (k, v) in attribute.items()]
        ret += "  %s [%s];\n" % (function_name(function), ",".join(attributes))
        # output functions
        for i, from_value in enumerate(function[b"inputs"]):
            ret += "  %s -> %s:a%d%s[constraint=false];" % (
                from_value.decode(),
                function_name(function),
                i,
                from_value.decode(),
            )
            ret += '  %s -> %s[color="transparent"];\n' % (from_value.decode(), function_name(function))
        for to_value in function[b"outputs"]:
            ret += "  %s:%s -> %s[constraint=false];" % (function_name(function), to_value.decode(), to_value.decode())
            ret += '  %s -> %s[color="transparent"];\n' % (function_name(function), to_value.decode())

    ret += "  { rank = same; \n"
    for var in inputs:
        attribute = {"xlabel": var.decode(), "shape": "point"}
        attributes = ['%s="%s"' % (k, v) for (k, v) in attribute.items()]
        ret += "    %s;\n" % (var.decode())
    ret += "  }\n"
    ret += "  { rank = same; \n"
    for var in outputs:
        attribute = {"xlabel": var.decode(), "shape": "point"}
        attributes = ['%s="%s"' % (k, v) for (k, v) in attribute.items()]
        ret += "    %s;\n" % (var.decode())
    ret += "  }\n"
    ret += "}"
    return ret


def nnoir2dot():
    parser = argparse.ArgumentParser(description="NNOIR to Graphviz(dot) Converter")
    parser.add_argument(dest="input", type=str, metavar="NNOIR", help="input(NNOIR) file path")
    args = parser.parse_args()
    with open(args.input, "rb") as f:
        print(to_dot(msgpack.unpackb(f.read(), raw=True)))
