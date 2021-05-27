import os

from setuptools import find_packages, setup

exec(open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "nnoir_onnx", "_version.py")).read())

with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md"),
    encoding="utf-8",
) as f:
    long_description = f.read()

setup(
    name="nnoir-onnx",
    version=__version__,
    description="ONNX to NNOIR Converter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Idein/nnoir/tree/master/nnoir-onnx",
    author="Idein Inc.",
    author_email="n.ohkawa@idein.jp",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="nnoir machine learning onnx",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "msgpack",
        "onnx<1.9.0",
        "onnxruntime>=1.2.0",
        "nnoir",
        "protobuf>=3.8",
    ],
    scripts=["onnx2nnoir", "freeze_onnx"],
)
