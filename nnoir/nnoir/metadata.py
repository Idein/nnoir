import argparse

from .load import load


def read_name(nnoir) -> str:
    return nnoir.name.decode("utf-8")


def read_description(nnoir) -> str:
    return nnoir.description.decode("utf-8")


def read_generator_name(nnoir) -> str:
    return nnoir.generator_name.decode("utf-8")


def read_generator_version(nnoir) -> str:
    return nnoir.generator_version.decode("utf-8")


def write_name(nnoir, name: str):
    nnoir.name = name.encode(encoding="utf-8")


def write_description(nnoir, description: str) -> str:
    nnoir.description = description.encode(encoding="utf-8")


def print_metadata(nnoir):
    print(f"name = {read_name(nnoir)}")
    print(f"description = {read_description(nnoir)}")
    print(f"generator.name = {read_generator_name(nnoir)}")
    print(f"generator.version = {read_generator_version(nnoir)}")


def nnoir_metadata():
    parser = argparse.ArgumentParser(description="read/write NNOIR meta data")
    parser.add_argument(dest="input", type=str, metavar="NNOIR", help="input(NNOIR) file path")
    parser.add_argument(
        "--write-description", type=str, dest="new_description", metavar="<new description>", help="overwrite description"
    )
    parser.add_argument("--write-name", type=str, dest="new_name", metavar="<new name>", help="overwrite name")

    args = parser.parse_args()
    nnoir = load(args.input)

    if args.new_description is None and args.new_name is None:
        return print_metadata(nnoir)

    if args.new_description is not None:
        write_description(nnoir, args.new_description)
    if args.new_name is not None:
        write_name(nnoir, args.new_name)

    nnoir.dump(args.input)
