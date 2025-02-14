[![DOI](https://zenodo.org/badge/562988605.svg)](https://zenodo.org/badge/latestdoi/562988605)
[![License: GPL-3.0](https://img.shields.io:/github/license/pism/pypac)](https://opensource.org/licenses/GPL-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

# pism-ragis: A Reanalysis for the Greenland Ice Sheet (RAGIS)

Repository for work related to NASA ROSES award 80NSSC21K0748 (2021-24).

## Synopsis

The stability of the Greenland Ice Sheet in a warming climate is a critical societal concern. Predicting Greenland's contribution to sea level remains a challenge as historical simulations of the past decades show limited agreement with observations. In this project, we develop a data assimilation framework that combines sparse observations and the ice sheet model PISM to produce a reanalysis of the state of the Greenland Ice Sheet from 1980 to 2020 using probabilistic filtering methods.

## Installation

Get pism-ragis source from GitHub:

    $ git clone git@github.com:pism/pism-ragis.git
    $ cd pism-ragis

Optionally create Conda environment named *pism-ragis*:

    $ conda env create -f environment.yml
    $ conda activate pism-ragis

or using Mamba instead:

    $ mamba env create -f environment.yml
    $ mamba activate pism-ragis

Install pism-ragis:

    $ pip install .


## Workflow

We strive to make our workflow reproducible, however, perfect reproducability is hard to achieve. For example, third-party providers of input data sets may change their access patterns, breaking the workflow. PISM needs to be installed on an HPC system with ample disk space (10s of TB).

### PISM

Please install PISM [![Parallel Ice Sheet Model (PISM)]](https://pism.io)
