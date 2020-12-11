#! /usr/bin/env python

from secrets import randbelow

# return random number in [min, max)
def get_rand(min, max):
    return randbelow(max - min) + min

# return (bit_length) bits random number
def get_rand_bits(bit_length):
    return get_rand(pow(2, bit_length - 1), pow(2, bit_length))
