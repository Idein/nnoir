import os
import setuptools
import blackonnxlib

exec(open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'blackonnx', '_version.py')).read())

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name=blackonnxlib,
    version=__version__,
    description="Adapt ONNX models to enable nnoir conversion",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Idein/nnoir/tree/master/blackonnx',
    author='Idein Inc.',
    author_email='christian@idein.jp',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords=blackonnxlib.DESCRIPTION,
    packages=setuptools.find_packages(),
    python_requires='>=3.6.*',
    install_requires=[
        'numpy',
        'onnx',
        'nnoir-onnx'
    ],
    scripts=['blackonnx']
)
