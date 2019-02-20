from setuptools import setup, find_packages
import os

exec(open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'nnoir', '_version.py')).read())

setup(
    name='nnoir',
    version=__version__,
    description='API for NNOIR',
    author='Idein Inc.',
    author_email='fujii@idein.jp',
    license='MIT',
    keywords='nnoir machine learning',
    packages=find_packages(),
    install_requires=['numpy', 'msgpack-python'],
    scripts=['nnoir2dot']
)
