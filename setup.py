#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="explicit_env",
    version="0.0.1",
    install_requires=["gym >= 0.2.3", "numpy", "numba", "python-interface",],
    packages=find_packages(),
)
