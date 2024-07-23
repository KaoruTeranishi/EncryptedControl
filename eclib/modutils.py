#! /usr/bin/env python3

from math import floor


def min_residue(a: int, m: int) -> int:
    return a - floor(a / m + 0.5) * m
