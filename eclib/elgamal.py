#! /usr/bin/env python

from eclib.numutils import *
from eclib.randutils import *
from eclib.primeutils import *
from eclib.modutils import *
from collections import namedtuple
import numpy as np
from math import floor, ceil

def keygen(bit_length):
    params = namedtuple('Parameters', ['p', 'q', 'g'])
    params.q, params.p = get_safe_prime(bit_length)
    params.g = get_generator(params.q, params.p)

    sk = get_rand(1, params.q)

    pk = mpow(params.g, sk, params.p)
    
    return params, pk, sk

def encrypt(params, pk, m):
    # scalar
    if isinstance(m, int):
        return _encrypt(params, pk, m)
    # vector
    elif isinstance(m[0], int):
        c = [[0, 0] for i in range(len(m))]
        for i in range(len(c)):
            c[i] = _encrypt(params, pk, m[i])
        return c
    # matrix
    elif isinstance(m[0][0], int):
        c = [[[0, 0] for j in range(len(m[0]))] for i in range(len(m))]
        for i in range(len(c)):
            for j in range(len(c[0])):
                c[i][j] = _encrypt(params, pk, m[i][j])
        return c
    else:
        print('error: encryption')
        return None

def decrypt(params, sk, c):
    # scalar
    if isinstance(c[0], int):
        return _decrypt(params, sk, c)
    # vector
    elif isinstance(c[0][0], int):
        m = [0 for i in range(len(c))]
        for i in range(len(m)):
            m[i] = _decrypt(params, sk, c[i])
        return m
    # matrix
    elif isinstance(c[0][0][0], int):
        m = [[0 for j in range(len(c[0]))] for i in range(len(c))]
        for i in range(len(m)):
            for j in range(len(m[0])):
                m[i][j] = _decrypt(params, sk, c[i][j])
        return m
    else:
        print('error: decryption')
        return None

def mult(params, c1, c2):
    # scalar x scalar
    if isinstance(c1[0], int) and isinstance(c2[0], int):
        return _mult(params, c1, c2)
    # scalar x vector
    if isinstance(c1[0], int) and isinstance(c2[0][0], int):
        c = [[0, 0] for i in range(len(c2))]
        for i in range(len(c)):
            c[i] = _mult(params, c1, c2[i])
        return c
    # scalar x matrix
    if isinstance(c1[0], int) and isinstance(c2[0][0][0], int):
        c = [[[0, 0] for j in range(len(c2[0]))] for i in range(len(c2))]
        for i in range(len(c)):
            for j in range(len(c[0])):
                c[i][j] = _mult(params, c1, c2[i][j])
        return c
    # vector x vector
    elif isinstance(c1[0][0], int) and isinstance(c2[0][0], int) and len(c1) == len(c2):
        c = [[0, 0] for i in range(len(c1))]
        for i in range(len(c)):
            c[i] = _mult(params, c1[i], c2[i])
        return c
    # matrix x vector
    elif isinstance(c1[0][0][0], int) and isinstance(c2[0][0], int) and len(c1[0]) == len(c2):
        c = [[[0, 0] for j in range(len(c1[0]))] for i in range(len(c1))]
        for i in range(len(c)):
            for j in range(len(c[0])):
                c[i][j] = _mult(params, c1[i][j], c2[j])
        return c
    # matrix x matrix
    elif isinstance(c1[0][0][0], int) and isinstance(c2[0][0][0], int) and len(c1) == len(c2) and len(c1[0]) == len(c2[0]):
        c = [[[0, 0] for j in range(len(c1[0]))] for i in range(len(c1))]
        for i in range(len(c)):
            for j in range(len(c[0])):
                c[i][j] = _mult(params, c1[i][j], c2[i][j])
        return c
    else:
        print('error: multiplication')
        return None

def encode(params, x, delta, mode='nearest'):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    # scalar
    if isinstance(x, float) or isinstance(x, int):
        return _encode(params, x, delta, mode)
    # vector
    elif isinstance(x[0], float) or isinstance(x[0], int):
        m = [0 for i in range(len(x))]
        for i in range(len(m)):
            m[i] = _encode(params, x[i], delta, mode)
        return m
    # matrix
    elif isinstance(x[0][0], float) or isinstance(x[0][0], int):
        m = [[0 for j in range(len(x[0]))] for i in range(len(x))]
        for i in range(len(m)):
            for j in range(len(m[0])):
                m[i][j] = _encode(params, x[i][j], delta, mode)
        return m
    else:
        print('error: encoding')
        return None

def decode(params, m, delta):
    # scalar
    if isinstance(m, int):
        return _decode(params, m, delta)
    # vector
    elif isinstance(m[0], int):
        x = [0 for i in range(len(m))]
        for i in range(len(x)):
            x[i] = _decode(params, m[i], delta)
        return x
    # matrix
    elif isinstance(m[0][0], int):
        x = [[0 for j in range(len(m[0]))] for i in range(len(m))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                x[i][j] = _decode(params, m[i][j], delta)
        return x
    else:
        print('error: decoding')
        return None

def enc(params, pk, x, delta, mode='nearest'):
    return encrypt(params, pk, encode(params, x, delta, mode))

def dec(params, sk, c, delta):
    return decode(params, decrypt(params, sk, c), delta)

def dec_add(params, sk, c, delta):
    # scalar
    if isinstance(c[0], int):
        return dec(params, sk, c, delta)
    # vector
    elif isinstance(c[0][0], int):
        x = 0
        for i in range(len(c)):
            x += dec(params, sk, c[i], delta)
        return x
    # matrix
    elif isinstance(c[0][0][0], int):
        x = [0 for i in range(len(c))]
        for i in range(len(c)):
            x[i] = dec(params, sk, c[i][0], delta)
            for j in range(1, len(c[0])):
                x[i] += dec(params, sk, c[i][j], delta)
        return x
    else:
        print('error: decryption')
        return None

def _encrypt(params, pk, m):
    r = get_rand(1, params.q)
    return [mpow(params.g, r, params.p), (m * mpow(pk, r, params.p)) % params.p]

def _decrypt(params, sk, c):
    return (minv(mpow(c[0], sk, params.p), params.p) * c[1]) % params.p

def _mult(params, c1, c2):
    return [(c1[0] * c2[0]) % params.p, (c1[1] * c2[1]) % params.p]

def _encode(params, x, delta, mode):
    if mode == 'nearest':
        m = floor(x / delta + 0.5)
        first_decimal_place = (x / delta * 10) % 10

        if m < 0:
            if m < -params.q:
                print('error: underflow')
                return None
            else:
                m += params.p
        elif m > params.q:
            print('error: overflow')
            return None

        if x / delta == int(x / delta) or first_decimal_place >= 5:
            for i in range(params.q):
                if m - i > 0 and is_element(m - i, params.q, params.p):
                    return m - i
                elif m + i < params.p and is_element(m + i, params.q, params.p):
                    return m + i
        else:
            for i in range(params.q):
                if m + i < params.p and is_element(m + i, params.q, params.p):
                    return m + i
                elif m - i > 0 and is_element(m - i, params.q, params.p):
                    return m - i
            
            print('error: encoding')
            return None
    
    elif mode == 'lower':
        m = ceil(x / delta)

        if m <= 0:
            if m < -params.q:
                print('error: underflow')
                return None
            else:
                m += params.p
        elif m > params.q:
            print('error: overflow')
            return None

        for i in range(m):
            if is_element(m - i, params.q, params.p):
                return m - i
        
        print('error: encoding')
        return None
    
    elif mode == 'upper':
        m = ceil(x / delta)

        if m < 0:
            if m < -params.q:
                print('error: underflow')
                return None
            else:
                m += params.p
        elif m > params.q:
            print('error: overflow')
            return None

        for i in range(params.p - m):
            if is_element(m + i, params.q, params.p):
                return m + i

        print('error: encoding')
        return None

    else:
        print('error: encoding')
        return None

def _decode(params, m, delta):
    return (m - params.p) * delta if m > params.q else m * delta
