#! /usr/bin/env python

from secrets import randbelow
import numpy as np
from math import floor

# return random number in [min, max)
def get_rand(min, max):
    return randbelow(max - min) + min

# return (bit_length) bits random number
def get_rand_bits(bit_length):
    return get_rand(pow(2, bit_length - 1), pow(2, bit_length))

def get_int_gaussian(mean, std, dim=1):
    if dim == 1:
        return floor(np.random.normal(mean, std) + 0.5)
    else:
        return [floor(r + 0.5) for r in np.random.normal(mean, std, dim)]
