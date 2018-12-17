from setuptools import setup, find_packages
setup(
    name="mlir2chainer",
    version="0.1",
    description='MLIR to Chainer Model Converter',
    author='Idein Inc.',
    author_email='fujii@idein.jp',
    license='MIT',
    keywords='mlir machine learning',
    packages=['chainer.links.mlir',
              'chainer.links.mlir.converter'],
    install_requires=['chainer', 'msgpack-python'],
)
