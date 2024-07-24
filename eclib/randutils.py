#! /usr/bin/env python3

from math import floor
from secrets import randbelow

import numpy as np


# return random number in [min, max)
def get_rand(min: int, max: int) -> int:
    return randbelow(max - min) + min


# return (bit_length) bits random number
def get_rand_bits(bit_length: int) -> int:
    return get_rand(pow(2, bit_length - 1), pow(2, bit_length))


def get_int_gaussian(mean: int, std: float, dim: int = 1) -> int | list[int]:
    if dim == 1:
        return floor(np.random.normal(mean, std) + 0.5)
    else:
        return [floor(r + 0.5) for r in np.random.normal(mean, std, dim)]
