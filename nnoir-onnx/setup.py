from setuptools import setup, find_packages
import os

exec(open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'nnoir_onnx', '_version.py')).read())

setup(
    name="nnoir-onnx",
    version=__version__,
    description='ONNX to NNOIR Converter',
    author='Idein Inc.',
    author_email='n.ohkawa@idein.jp',
    license='MIT',
    keywords='nnoir machine learning onnx',
    packages=find_packages(),
    install_requires=['numpy', 'msgpack-python', 'onnx', 'onnxruntime', 'nnoir'],
    scripts=['onnx2nnoir']
)
