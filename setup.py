#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: skip-file

from pathlib import Path

from setuptools import setup

with open(str(Path(".", "VERSION").absolute())) as version_file:
    version = version_file.read().strip()

PKG_NAME = "pismanalysis"

packages = [PKG_NAME]

# Dependencies of pism-analysis
requirements = ["pyproj", "configobj", "jinja2"]

setup(
    name="pismanalysis",
    version=version,
    author="Andy Aschwanden",
    author_email="andy.aschwanden@gmail.com",
    description=("Postprocessing for PISM"),
    license="GPL 3.0",
    keywords="PISM",
    url="https://github.com/pism/pism-analysis",
    project_urls={
        "Bug Tracker": "https://github.com/pism/pism-analysis/issues",
        "Documentation": "https://github.com/pism/pism-analysis",
        "Source Code": "https://github.com/pism/pism-analysis",
    },
    packages=[PKG_NAME],
    package_dir={PKG_NAME: PKG_NAME},
    install_requires=requirements,
    extras_require={
        "develop": [
            "flake8",
            "pytest",
        ]
    },
    entry_points={
        "console_scripts": [
            f"extract_profiles = {PKG_NAME}.extract_profiles:main",
        ]
    },
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
