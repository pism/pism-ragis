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

import logging
import time
import tracemalloc
from functools import wraps

from pism_ragis.logger import get_logger

logger: logging.Logger = get_logger("pism_ragis")


def timeit(func):
    """
    Decorator that logs the time a function takes to execute.

    This decorator logs the start time, end time, and the elapsed time
    for the execution of the decorated function.

    Parameters
    ----------
    func : callable
        The function to be decorated.

    Returns
    -------
    callable
        The wrapped function with added timing functionality.

    Examples
    --------
    >>> @timeit
    ... def example_function():
    ...     import time; time.sleep(1)
    ...
    >>> example_function()
    INFO:__main__:example_function: Starting
    INFO:__main__:example_function: Finished in 1.00 seconds
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapper function that logs the execution time of the decorated function.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to the decorated function.
        **kwargs : dict
            Keyword arguments to pass to the decorated function.

        Returns
        -------
        Any
           The result of the decorated function.
        """
        start_time = time.time()
        logger.info("%s: Starting", func.__name__)
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info("%s: Finished in %2.2f seconds", func.__name__, elapsed_time)
        return result

    return wrapper


def profileit(func):
    """
    Decorator that logs the time and memory usage of a function.

    This decorator logs the start time, end time, elapsed time, current memory usage,
    and peak memory usage for the execution of the decorated function.

    Parameters
    ----------
    func : callable
        The function to be decorated.

    Returns
    -------
    callable
        The wrapped function with added profiling functionality.

    Examples
    --------
    >>> @profileit
    ... def example_function():
    ...     time.sleep(1)
    ...
    >>> example_function()
    INFO:__main__:Starting example_function
    INFO:__main__:example_function: Memory usage: 0.10 MB
    INFO:__main__:example_function: Peak memory usage: 0.15 MB
    INFO:__main__:example_function Finished in 1.00 seconds
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapper function that logs the execution time and memory usage of the decorated function.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to the decorated function.
        **kwargs : dict
            Keyword arguments to pass to the decorated function.

        Returns
        -------
        Any
            The result of the decorated function.
        """
        tracemalloc.start()
        start_time = time.time()
        logger.info("Starting %s", func.__name__)
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info("%s: Memory usage: %.2f MB", func.__name__, current / 1024**2)
        logger.info("%s: Peak memory usage: %.2f MB", func.__name__, peak / 1024**2)
        logger.info("%s Finished in %.2f seconds", func.__name__, elapsed_time)
        return result

    return wrapper
