#! /usr/bin/env python3


def min_residue(a: int, m: int) -> int:
    b = a % m
    c = b - m
    if b >= abs(c):
        return c
    else:
        return b
