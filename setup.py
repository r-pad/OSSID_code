#!/usr/bin/env python
from distutils.core import setup, Extension
from setuptools import find_packages

setup(
    name='ossid',
    version='1.0',
    author='Qiao Gu',
    packages=find_packages('python'),
    package_dir={'': 'python'},
    description= 'Python source code for OSSID',
    long_description='',
    package_data={'': ['*.so']}
)

