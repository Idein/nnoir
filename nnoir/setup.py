import os

from setuptools import find_packages, setup

exec(open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "nnoir", "_version.py")).read())

with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md"),
    encoding="utf-8",
) as f:
    long_description = f.read()

setup(
    name="nnoir",
    version=__version__,
    description="API for NNOIR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Idein/nnoir/tree/master/nnoir",
    author="Idein Inc.",
    author_email="fujii@idein.jp",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="nnoir machine learning",
    packages=find_packages(),
    install_requires=["numpy", "msgpack"],
    scripts=["nnoir2dot", "nnrunner"],
)
