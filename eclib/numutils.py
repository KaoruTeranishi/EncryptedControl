#! /usr/bin/env python3

from math import gcd

import eclib.modutils as mu


def lcm(a, b):
    return (a * b) // gcd(a, b)


def is_generator(g, q, p):
    if g <= 1 or g >= p:
        return False
    elif mu.mpow(g, q, p) == 1:
        return True
    else:
        return False


def is_element(m, q, p):
    if m <= 0 or m >= p:
        return False
    elif mu.mpow(m, q, p) == 1:
        return True
    else:
        return False


def get_generator(q, p):
    g = 2
    while not is_generator(g, q, p):
        g += 1

    return g


def get_dmax(params):
    G = [mu.mpow(params.g, i, params.p) for i in range(params.q)]
    G.sort()
    G.append(params.p)
    return max([x - y for (x, y) in zip(G[1:], G[:-1])])
