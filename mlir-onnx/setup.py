from setuptools import setup, find_packages
import os

exec(open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'mlir_onnx', '_version.py')).read())

setup(
    name="mlir-onnx",
    version=__version__,
    description='ONNX to MLIR Converter',
    author='Idein Inc.',
    author_email='n.ohkawa@idein.jp',
    license='MIT',
    keywords='mlir machine learning onnx',
    packages=find_packages(),
    install_requires=['numpy', 'msgpack-python', 'onnx', 'onnxruntime'],
    scripts=['onnx2mlir']
)
