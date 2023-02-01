#! /usr/bin/env python

from eclib.modutils import *
from math import gcd

def lcm(a, b):
    return (a * b) // gcd(a, b)

def is_generator(g, q, p):
    if g <= 1 or g >= p:
        return False
    elif mpow(g, q, p) == 1:
        return True
    else:
        return False

def is_element(m, q, p):
    if m <= 0 or m >= p:
        return False
    elif mpow(m, q, p) == 1:
        return True
    else:
        return False

def get_generator(q, p):
    g = 2
    while not is_generator(g, q, p):
        g += 1
    
    return g

def get_dmax(params):
    G = [mpow(params.g, i, params.p) for i in range(params.q)]
    G.sort()
    G.append(params.p)
    return max([x - y for (x, y) in zip(G[1:], G[:-1])])
