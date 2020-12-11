#! /usr/bin/env python

from eclib.numutils import *
from eclib.randutils import *
from eclib.primeutils import *
from eclib.modutils import *
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
        c = [1 for i in range(len(m))]
        for i in range(len(c)):
            c[i] = _encrypt(params, pk, m[i])
        return c
    # matrix
    elif isinstance(m[0][0], int):
        c = [[1 for j in range(len(m[0]))] for i in range(len(m))]
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
        m = [0 for i in range(len(c))]
        for i in range(len(m)):
            m[i] = _decrypt(params, sk, c[i])
        return m
    # matrix
    elif isinstance(c[0][0], int):
        m = [[0 for j in range(len(c[0]))] for i in range(len(c))]
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
        c = [1 for i in range(len(c1))]
        for i in range(len(c)):
            c[i] = _add(params, c1[i], c2[i])
        return c
    # matrix + matrix
    elif isinstance(c1[0][0], int) and isinstance(c2[0][0], int) and len(c1) == len(c2) and len(c1[0]) == len(c2[0]):
        c = [[1 for j in range(len(c1[0]))] for i in range(len(c1))]
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
        c = [[1 for j in range(len(c1[0]))] for i in range(len(c1))]
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

def elementwise_mult(params, m, c):
    # scalar (plaintext) x scalar (ciphertext)
    if isinstance(m, int) and isinstance(c, int):
        return _mult(params, m, c)
    # vector (plaintext) x vector (ciphertext)
    elif isinstance(m[0], int) and isinstance(c[0], int) and len(m) == len(c):
        c_ = [1 for i in range(len(m))]
        for i in range(len(c_)):
            c_[i] = _mult(params, m[i], c[i])
        return c_
    # matrix (plaintext) x vector (ciphertext)
    elif isinstance(m[0][0], int) and isinstance(c[0], int) and len(m[0]) == len(c):
        c_ = [[1 for j in range(len(m[0]))] for i in range(len(m))]
        for i in range(len(c_)):
            for j in range(len(c_[0])):
                c_[i][j] = _mult(params, m[i][j], c[j])
        return c_
    # matrix (plaintext) x matrix (ciphertext)
    elif isinstance(m[0][0], int) and isinstance(c[0][0], int) and len(m) == len(c) and len(m[0]) == len(c[0]):
        c_ = [[1 for j in range(len(m[0]))] for i in range(len(m))]
        for i in range(len(c_)):
            for j in range(len(c_[0])):
                c_[i][j] = _mult(params, m[i][j], c[i][j])
        return c_
    else:
        print('error: elementwise multiplication')
        return None

def mult(params, m, c):
    # scalar (plaintext) x scalar (ciphertext)
    if isinstance(m, int) and isinstance(c, int):
        return _mult(params, m, c)
    # scalar (plaintext) x vector (ciphertext)
    elif isinstance(m, int) and isinstance(c[0], int):
        c_ = [1 for i in range(len(c))]
        for i in range(len(c)):
            c_[i] = _mult(params, m, c[i])
        return c_
    # scalar (plaintext) x matrix (ciphertext)
    elif isinstance(m, int) and isinstance(c[0][0], int):
        c_ = [[1 for j in range(len(c[0]))] for i in range(len(c))]
        for i in range(len(c)):
            for j in range(len(c[0])):
                c_[i][j] = _mult(params, m, c[i][j])
        return c_
    # vector (plaintext) x vector (ciphertext)
    elif isinstance(m[0], int) and isinstance(c[0], int) and len(m) == len(c):
        c_ = 1
        for i in range(len(m)):
            c_ = _add(params, c_, _mult(params, m[i], c[i]))
        return c_
    # matrix (plaintext) x vector (ciphertext)
    elif isinstance(m[0][0], int) and isinstance(c[0], int) and len(m[0]) == len(c):
        c_ = [1 for i in range(len(m))]
        for i in range(len(m)):
            for j in range(len(c)):
                c_[i] = _add(params, c_[i], _mult(params, m[i][j], c[j]))
        return c_
    # matrix (plaintext) x matrix (ciphertext)
    elif isinstance(m[0][0], int) and isinstance(c[0][0], int) and len(m[0]) == len(c):
        c_ = [[1 for j in range(len(c[0]))] for i in range(len(m))]
        for i in range(len(m)):
            for j in range(len(c[0])):
                for k in range(len(m[0])):
                    c_[i][j] = _add(params, c_[i][j], _mult(params, m[i][k], c[k][j]))
        return c_
    else:
        print('error: multiplication')
        return None

def encode(x, a, b, a_, b_):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    # scalar
    if isinstance(x, float) or isinstance(x, int):
        return _encode(x, a, b, a_, b_)
    # vector
    elif isinstance(x[0], float) or isinstance(x[0], int):
        m = [0 for i in range(len(x))]
        for i in range(len(m)):
            m[i] = _encode(x[i], a, b, a_, b_)
        return m
    # matrix
    elif isinstance(x[0][0], float) or isinstance(x[0][0], int):
        m = [[0 for j in range(len(x[0]))] for i in range(len(x))]
        for i in range(len(m)):
            for j in range(len(m[0])):
                m[i][j] = _encode(x[i][j], a, b, a_, b_)
        return m
    else:
        print('error: encoding')
        return None

def decode(m, a, b):
    # scalar
    if isinstance(m, int):
        return _decode(m, a, b)
    # vector
    elif isinstance(m[0], int):
        x = [0 for i in range(len(m))]
        for i in range(len(x)):
            x[i] = _decode(m[i], a, b)
        return x
    # matrix
    elif isinstance(m[0][0], int):
        x = [[0 for j in range(len(m[0]))] for i in range(len(m))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                x[i][j] = _decode(m[i][j], a, b)
        return x
    else:
        print('error: decoding')
        return None

def decode_(m, a, b):
    # scalar
    if isinstance(m, int):
        return _decode_(m, a, b)
    # vector
    elif isinstance(m[0], int):
        x = [0 for i in range(len(m))]
        for i in range(len(x)):
            x[i] = _decode_(m[i], a, b)
        return x
    # matrix
    elif isinstance(m[0][0], int):
        x = [[0 for j in range(len(m[0]))] for i in range(len(m))]
        for i in range(len(x)):
            for j in range(len(x[0])):
                x[i][j] = _decode_(m[i][j], a, b)
        return x
    else:
        print('error: decoding')
        return None

def enc(params, pk, x, a, b, a_, b_):
    return encrypt(params, pk, encode(x, a, b, a_, b_))

def dec(params, sk, c, a, b):
    return decode(decrypt(params, sk, c), a, b)

def dec_(params, sk, c, a, b):
    return decode_(decrypt(params, sk, c), a, b)

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

def _mult(params, m, c):
    return mpow(c, m, params.n_square)

def _encode(x, a, b, a_, b_):
    if x < -pow(2, a_):
        print('error: underflow')
        return None
    elif x > pow(2, a_) - pow(2, -b_):
        print('error: overflow')
        return None
    else:
        q = floor(x * pow(2, b_) + 0.5) / pow(2, b_) # encoding to fixed point number
        return int((pow(2, b) * q) % pow(2, a + 2 * b))

def _decode(m, a, b):
    return min_residue(m, pow(2, a + 2 * b)) / pow(2, 2 * b)

def _decode_(m, a, b):
    return min_residue(m, pow(2, a + 2 * b)) / pow(2, b)
