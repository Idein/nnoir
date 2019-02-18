from setuptools import setup, find_packages
setup(
    name='mlir',
    version='1.0.0',
    description='API for MLIR',
    author='Idein Inc.',
    author_email='fujii@idein.jp',
    license='MIT',
    keywords='mlir machine learning',
    packages=find_packages(),
    install_requires=['numpy', 'msgpack-python'],
    scripts=['mlir2dot']
)
