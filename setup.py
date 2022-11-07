#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: skip-file

from pathlib import Path

from setuptools import setup

with open(str(Path(".", "VERSION").absolute())) as version_file:
    version = version_file.read().strip()

packages = ["pypac"]

# Dependencies of pism-emulator
requirements = ["configobj", "jinja2", "pre-commit"]

setup(
    name="pypac",
    version=version,
    author="Andy Aschwanden",
    author_email="andy.aschwanden@gmail.com",
    description=("Postprocessing for PISM"),
    license="GPL 3.0",
    keywords="PISM",
    url="https://github.com/pism/pypac",
    project_urls={
        "Bug Tracker": "https://github.com/pism/pypac/issues",
        "Documentation": "https://github.com/pism/pypac",
        "Source Code": "https://github.com/pism/pypac",
    },
    packages=packages,
    install_requires=requirements,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Postprocessing",
    ],
)
