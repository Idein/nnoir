import argparse
from typing import Optional

import msgpack


def read_name(nnoir) -> str:
    return nnoir[b"nnoir"][b"model"][b"name"].decode("ascii")


def read_description(nnoir) -> str:
    description = ""
    if b"description" in nnoir[b"nnoir"][b"model"]:
        description = nnoir[b"nnoir"][b"model"][b"description"].decode("ascii")

    return description


def read_generator_name(nnoir) -> str:
    return nnoir[b"nnoir"][b"model"][b"generator"][b"name"].decode("ascii")


def read_generator_version(nnoir) -> str:
    return nnoir[b"nnoir"][b"model"][b"generator"][b"version"].decode("ascii")


def write_name(nnoir, name: str):
    name = nnoir[b"nnoir"][b"model"][b"name"] = name.encode(encoding="utf-8")


def write_description(nnoir, description: str) -> str:
    nnoir[b"nnoir"][b"model"][b"description"] = description.encode(encoding="utf-8")


def write_generator_name(nnoir, generator_name) -> str:
    generator_name = nnoir[b"nnoir"][b"model"][b"generator"][b"name"] = generator_name.encode(encoding="utf-8")


def write_generator_version(nnoir, generator_version):
    generator_version = nnoir[b"nnoir"][b"model"][b"generator"][b"version"] = generator_version.encode(encoding="utf-8")


def print_metadata(nnoir):
    print(f"name = {read_name(nnoir)}")
    print(f"description = {read_description(nnoir)}")
    print(f"generator.name = {read_generator_name(nnoir)}")
    print(f"generator.version = {read_generator_version(nnoir)}")


def read_metadata(nnoir, key: str) -> str:
    if key == "name":
        return read_name(nnoir)
    elif key == "description":
        return read_description(nnoir)
    elif key == "generator.name":
        return read_generator_name(nnoir)
    elif key == "generator.version":
        return read_generator_version(nnoir)
    else:
        raise RuntimeError(f"invalid key for metadat: {key}")


def write_metadata(nnoir, key: str, value: str):
    if key == "name":
        write_name(nnoir, value)
    elif key == "description":
        write_description(nnoir, value)
    elif key == "generator.name":
        write_generator_name(nnoir, value)
    elif key == "generator.version":
        write_generator_version(nnoir, value)


def main(file_name: str, key: Optional[str], value: Optional[str]):
    with open(file_name, "rb") as f:
        nnoir = msgpack.unpackb(f.read(), raw=True)

        if key is None:
            return print_metadata(nnoir)

        if value is None:
            return print(read_metadata(nnoir, key))

        write_metadata(nnoir, key, value)

    with open(file_name, "wb") as f:
        f.write(msgpack.packb(nnoir, use_bin_type=False))


def nnoir_metadata():
    parser = argparse.ArgumentParser(description="edit NNOIR meta data")
    parser.add_argument(dest="input", type=str, metavar="NNOIR", help="input(NNOIR) file path")
    parser.add_argument(dest="key", type=str, nargs="?", metavar="<key>", help="meta data key")
    parser.add_argument(dest="value", type=str, nargs="?", metavar="<value>", help="meta data value")
    args = parser.parse_args()

    main(args.input, args.key, args.value)
