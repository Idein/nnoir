from setuptools import setup, find_packages
setup(
    name="mlir-onnx",
    version="1.0.0",
    description='ONNX to MLIR Converter',
    author='Idein Inc.',
    author_email='n.ohkawa@idein.jp',
    license='MIT',
    keywords='mlir machine learning onnx',
    packages=find_packages(),
    install_requires=['numpy', 'msgpack-python', 'onnx', 'onnxruntime'],
    scripts=['onnx2mlir']
)
