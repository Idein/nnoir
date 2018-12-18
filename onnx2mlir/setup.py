from setuptools import setup, find_packages
setup(
    name="onnx2mlir",
    version="0.1",
    description='ONNX to MLIR Converter',
    author='Idein Inc.',
    author_email='n.ohkawa@idein.jp',
    license='MIT',
    keywords='mlir machine learning onnx',
    packages=['mlir.onnx','mlir.onnx.operators'],
    install_requires=['numpy', 'msgpack-python', 'onnx', 'onnxruntime', 'mlir'],
    scripts=['onnx2mlir']
)
