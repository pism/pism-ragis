# Copyright (C) 2024 Andy Aschwanden
#
# This file is part of pism-ragis.
#
# PISM-RAGIS is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PISM-RAGIS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software

# pylint: disable=too-many-positional-arguments

"""
Module for handling logging.
"""

import importlib.resources
import logging


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name and configure it using a logging configuration file.

    This function retrieves a logger with the specified name, configures it using a logging
    configuration file located within the package, and disables logging for the "matplotlib" logger.

    Parameters
    ----------
    name : str
        The name of the logger.

    Returns
    -------
    logging.Logger
        The configured logger object.

    Examples
    --------
    >>> logger = get_logger("my_logger")
    >>> logger.info("This is an info message.")
    """
    logger: logging.Logger = logging.getLogger(name)
    config_path = importlib.resources.files("pism_ragis").joinpath("logging.conf")
    with importlib.resources.as_file(config_path) as file_path:
        logging.config.fileConfig(file_path)  # type: ignore [attr-defined]

    logging.getLogger("matplotlib").disabled = True
    return logger
