#! /usr/bin/env python

# Bruce Schneier algorithm
def mpow(a, b, m):
    if a < 0:
        a += m

    c = 1
    while b >= 1:
        if b % 2 == 1:
            c = (a * c) % m
        a = pow(a, 2) % m
        b = b // 2

    return c

# extended Euclidean algorithm
def minv(a, m):
    b = m
    u = 1
    v = 0

    while b != 0:
        t = a // b
        a = a - t * b
        u = u - t * v
        a, b = b, a
        u, v = v, u
    
    return u + m if u < 0 else u

def min_residue(a, m):
    b = a % m
    c = b - m
    if b >= abs(c):
        return c
    else:
        return b
