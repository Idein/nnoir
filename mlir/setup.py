from setuptools import setup, find_packages
import os

exec(open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'mlir', '_version.py')).read())

setup(
    name='mlir',
    version=__version__,
    description='API for MLIR',
    author='Idein Inc.',
    author_email='fujii@idein.jp',
    license='MIT',
    keywords='mlir machine learning',
    packages=find_packages(),
    install_requires=['numpy', 'msgpack-python'],
    scripts=['mlir2dot']
)
