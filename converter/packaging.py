import os
import pypandoc

with open('README.rst','w+') as f:
    f.write(pypandoc.convert('README.md', 'rst'))
os.system("python setup.py sdist")
