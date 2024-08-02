#! /usr/bin/env python3

"""
randutils.py

This module provides utility functions for generating random numbers. The module
includes functions for generating random integers in a specified range, random integers
with a specified number of bits, and random integers sampled from a discrete Gaussian
distribution.

Functions:
    get_rand: Generate a random integer within a specified range.
    get_rand_bits: Generate a random integer with a specified number of bits.
    get_int_gaussian: Generate a random integer sampled from a discrete Gaussian
        distribution.

Dependencies:
    numpy: Fundamental package for scientific computing with Python.

Note:
    The module uses the secrets module for generating cryptographically strong random
    numbers.
"""

from math import floor
from secrets import randbelow

import numpy as np


# return random number in [min, max)
def get_rand(min: int, max: int) -> int:
    """
    Generates a random integer in [`min`, `max`).

    Args:
        min (int): The minimum value of the range (inclusive).
        max (int): The maximum value of the range (exclusive).

    Returns:
        int: The random integer in [`min`, `max`).
    """

    return randbelow(max - min) + min


def get_rand_bits(bit_length: int) -> int:
    """
    Generates a random integer with the specified number of bits.

    Args:
        bit_length (int): Desired bit length of the random integer.

    Returns:
        int: The random integer with `bit_length` bits.
    """

    return get_rand(pow(2, bit_length - 1), pow(2, bit_length))


def get_int_gaussian(mean: int, std: float, dim: int = 1) -> int | list[int]:
    """
    Generate a random integer or a list of random integers sampled from a discrete
    Gaussian distribution.

    Args:
        mean (int): Mean of the Gaussian distribution.
        std (float): Standard deviation of the Gaussian distribution.
        dim (int, optional, default = 1): Dimension of the output.

    Returns:
        int or list[int]: If `dim` is 1, returns a random integer sampled from the
        Gaussian distribution. Otherwise, returns a `dim`-dimensional list of random
        integers sampled from the Gaussian distribution.
    """

    if dim == 1:
        return floor(np.random.normal(mean, std) + 0.5)

    else:
        return [floor(r + 0.5) for r in np.random.normal(mean, std, dim)]
