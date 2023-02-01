#! /usr/bin/env python

from numutils import *
from randutils import *
from primeutils import *
from modutils import *
from collections import namedtuple
import numpy as np
from math import gcd, floor

def keygen(bit_length):
    p = get_prime(bit_length)
    q = get_prime(bit_length)
    while gcd(p * q, (p - 1) * (q - 1)) != 1 or p == q:
        p = get_prime(bit_length)
        q = get_prime(bit_length)
    params = namedtuple('Parameters', ['n', 'n_square'])
    params.n = p * q
    params.n_square = params.n * params.n

    k = get_rand(0, params.n)
    while gcd(k, params.n) != 1:
        k = get_rand(0, params.n)
    pk = (k * params.n + 1) % params.n_square

    sk = namedtuple('SecretKey', ['lmd', 'mu'])
    sk.lmd = lcm(p - 1, q - 1)
    sk.mu = minv(_L(mpow(pk, sk.lmd, params.n_square), params.n), params.n)

    return params, pk, sk

def encrypt(params, pk, m):
    # scalar
    if isinstance(m, int):
        return _encrypt(params, pk, m)
    # vector
    elif isinstance(m[0], int):
        c = [1 for _ in range(len(m))]
        for i in range(len(c)):
            c[i] = _encrypt(params, pk, m[i])
        return c
    # matrix
    elif isinstance(m[0][0], int):
        c = [[1 for _ in range(len(m[0]))] for _ in range(len(m))]
        for i in range(len(c)):
            for j in range(len(c[0])):
                c[i][j] = _encrypt(params, pk, m[i][j])
        return c
    else:
        print('error: encryption')
        return None

def decrypt(params, sk, c):
    # scalar
    if isinstance(c, int):
        return _decrypt(params, sk, c)
    # vector
    elif isinstance(c[0], int):
        m = [0 for _ in range(len(c))]
        for i in range(len(m)):
            m[i] = _decrypt(params, sk, c[i])
        return m
    # matrix
    elif isinstance(c[0][0], int):
        m = [[0 for _ in range(len(c[0]))] for _ in range(len(c))]
        for i in range(len(m)):
            for j in range(len(m[0])):
                m[i][j] = _decrypt(params, sk, c[i][j])
        return m
    else:
        print('error: decryption')
        return None

def add(params, c1, c2):
    # scalar + scalar
    if isinstance(c1, int) and isinstance(c2, int):
        return _add(params, c1, c2)
    # vector + vector
    elif isinstance(c1[0], int) and isinstance(c2[0], int) and len(c1) == len(c2):
        c = [1 for _ in range(len(c1))]
        for i in range(len(c)):
            c[i] = _add(params, c1[i], c2[i])
        return c
    # matrix + matrix
    elif isinstance(c1[0][0], int) and isinstance(c2[0][0], int) and len(c1) == len(c2) and len(c1[0]) == len(c2[0]):
        c = [[1 for _ in range(len(c1[0]))] for _ in range(len(c1))]
        for i in range(len(c)):
            for j in range(len(c[0])):
                c[i][j] = _add(params, c1[i][j], c2[i][j])
        return c
    else:
        print('error: addtion')
        return None

def elementwise_add(params, c1, c2):
    # scalar + scalar
    if isinstance(c1, int) and isinstance(c2, int):
        return add(params, c1, c2)
    # vector + vector
    elif isinstance(c1[0], int) and isinstance(c2[0], int) and len(c1) == len(c2):
        return add(params, c1, c2)
    # matrix + vector
    elif isinstance(c1[0][0], int) and isinstance(c2[0], int) and len(c1[0]) == len(c2):
        c = [[1 for _ in range(len(c1[0]))] for _ in range(len(c1))]
        for i in range(len(c)):
            for j in range(len(c[0])):
                c[i][j] = _add(params, c1[i][j], c2[j])
        return c
    # matrix + matrix
    elif isinstance(c1[0][0], int) and isinstance(c2[0][0], int) and len(c1) == len(c2) and len(c1[0]) == len(c2[0]):
        return add(params, c1, c2)
    else:
        print('error: elementwise addtion')
        return None

def int_mult(params, m, c):
    # scalar (plaintext) x scalar (ciphertext)
    if isinstance(m, int) and isinstance(c, int):
        return _int_mult(params, m, c)
    # scalar (plaintext) x vector (ciphertext)
    elif isinstance(m, int) and isinstance(c[0], int):
        c_ = [1 for _ in range(len(c))]
        for i in range(len(c)):
            c_[i] = _int_mult(params, m, c[i])
        return c_
    # scalar (plaintext) x matrix (ciphertext)
    elif isinstance(m, int) and isinstance(c[0][0], int):
        c_ = [[1 for _ in range(len(c[0]))] for _ in range(len(c))]
        for i in range(len(c)):
            for j in range(len(c[0])):
                c_[i][j] = _int_mult(params, m, c[i][j])
        return c_
    # vector (plaintext) x vector (ciphertext)
    elif isinstance(m[0], int) and isinstance(c[0], int) and len(m) == len(c):
        c_ = 1
        for i in range(len(m)):
            c_ = _add(params, c_, _int_mult(params, m[i], c[i]))
        return c_
    # matrix (plaintext) x vector (ciphertext)
    elif isinstance(m[0][0], int) and isinstance(c[0], int) and len(m[0]) == len(c):
        c_ = [1 for _ in range(len(m))]
        for i in range(len(m)):
            for j in range(len(c)):
                c_[i] = _add(params, c_[i], _int_mult(params, m[i][j], c[j]))
        return c_
    # matrix (plaintext) x matrix (ciphertext)
    elif isinstance(m[0][0], int) and isinstance(c[0][0], int) and len(m[0]) == len(c):
        c_ = [[1 for _ in range(len(c[0]))] for _ in range(len(m))]
        for i in range(len(m)):
            for j in range(len(c[0])):
                for k in range(len(m[0])):
                    c_[i][j] = _add(params, c_[i][j], _int_mult(params, m[i][k], c[k][j]))
        return c_
    else:
        print('error: integer multiplication')
        return None

def elementwise_int_mult(params, m, c):
    # scalar (plaintext) x scalar (ciphertext)
    if isinstance(m, int) and isinstance(c, int):
        return int_mult(params, m, c)
    # scalar (plaintext) x vector (ciphertext)
    elif isinstance(m, int) and isinstance(c[0], int):
        return int_mult(params, m, c)
    # scalar (plaintext) x matrix (ciphertext)
    elif isinstance(m, int) and isinstance(c[0][0], int):
        return int_mult(params, m, c)
    # vector (plaintext) x vector (ciphertext)
    elif isinstance(m[0], int) and isinstance(c[0], int) and len(m) == len(c):
        c_ = [1 for _ in range(len(m))]
        for i in range(len(c_)):
            c_[i] = _int_mult(params, m[i], c[i])
        return c_
    # matrix (plaintext) x vector (ciphertext)
    elif isinstance(m[0][0], int) and isinstance(c[0], int) and len(m[0]) == len(c):
        c_ = [[1 for _ in range(len(m[0]))] for _ in range(len(m))]
        for i in range(len(c_)):
            for j in range(len(c_[0])):
                c_[i][j] = _int_mult(params, m[i][j], c[j])
        return c_
    # matrix (plaintext) x matrix (ciphertext)
    elif isinstance(m[0][0], int) and isinstance(c[0][0], int) and len(m) == len(c) and len(m[0]) == len(c[0]):
        c_ = [[1 for _ in range(len(m[0]))] for _ in range(len(m))]
        for i in range(len(c_)):
            for j in range(len(c_[0])):
                c_[i][j] = _int_mult(params, m[i][j], c[i][j])
        return c_
    else:
        print('error: elementwise integer multiplication')
        return None

def encode(params, x, delta):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    # scalar
    if isinstance(x, float) or isinstance(x, int):
        return _encode(params, x, delta)
    # vector
    elif isinstance(x[0], float) or isinstance(x[0], int):
        m = [0 for _ in range(len(x))]
        for i in range(len(m)):
            m[i] = _encode(params, x[i], delta)
        return m
    # matrix
    elif isinstance(x[0][0], float) or isinstance(x[0][0], int):
        m = [[0 for j in range(len(x[0]))] for i in range(len(x))]
        for i in range(len(m)):
            for j in range(len(m[0])):
                m[i][j] = _encode(params, x[i][j], delta)
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
        x = [0 for _ in range(len(m))]
        for i in range(len(x)):
            x[i] = _decode(params, m[i], delta)
        return x
    # matrix
    elif isinstance(m[0][0], int):
        x = [[0 for _ in range(len(m[0]))] for _ in range(len(m))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                x[i][j] = _decode(params, m[i][j], delta)
        return x
    else:
        print('error: decoding')
        return None

def enc(params, pk, x, delta):
    return encrypt(params, pk, encode(params, x, delta))

def dec(params, sk, c, delta):
    return decode(params, decrypt(params, sk, c), delta)

def _L(x, n):
    return (x - 1) // n

def _encrypt(params, pk, m):
    r = get_rand(0, params.n)
    while gcd(r, params.n) != 1:
        r = get_rand(0, params.n)
    return (mpow(pk, m, params.n_square) * mpow(r, params.n, params.n_square)) % params.n_square

def _decrypt(params, sk, c):
    return (_L(mpow(c, sk.lmd, params.n_square), params.n) * sk.mu) % params.n

def _add(params, c1, c2):
    return (c1 * c2) % params.n_square

def _int_mult(params, m, c):
    return mpow(c, m, params.n_square)

def _encode(params, x, delta):
    m = floor(x / delta + 0.5)

    if m < 0:
        if m < -floor((params.n - 1) / 2):
            print('error: underflow')
            return None
        else:
            m += params.n
    elif m > floor(params.n / 2):
        print('error: overflow')
        return None

    return m

def _decode(params, m, delta):
    return min_residue(m, params.n) * delta
