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
    if isinstance(m, np.matrix):
        m = np.array(m)

    # scalar
    if isinstance(m, int):
        return _encrypt(params, pk, m)
    # vector
    elif isinstance(m[0], int):
        c = np.ones(m.shape, dtype=object)
        for i in range(c.shape[0]):
            c[i] = _encrypt(params, pk, m[i])
        return c
    # matrix
    elif isinstance(m[0][0], int):
        c = np.ones(m.shape, dtype=object)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                c[i][j] = _encrypt(params, pk, m[i][j])
        return c
    else:
        print('error: encryption')
        return None

def decrypt(params, sk, c):
    if isinstance(c, np.matrix):
        c = np.array(c)

    # scalar
    if isinstance(c, int):
        return _decrypt(params, sk, c)
    # vector
    elif isinstance(c[0], int):
        m = np.zeros(c.shape, dtype=object)
        for i in range(m.shape[0]):
            m[i] = _decrypt(params, sk, c[i])
        return m
    # matrix
    elif isinstance(c[0][0], int):
        m = np.zeros(c.shape, dtype=object)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                m[i][j] = _decrypt(params, sk, c[i][j])
        return m
    else:
        print('error: decryption')
        return None

def add(params, c1, c2):
    if isinstance(c1, np.matrix):
        c1 = np.array(c1)
    if isinstance(c2, np.matrix):
        c2 = np.array(c2)

    # scalar + scalar
    if isinstance(c1, int) and isinstance(c2, int):
        return _add(params, c1, c2)
    # vector + vector
    elif isinstance(c1[0], int) and isinstance(c2[0], int) and c1.shape == c2.shape:
        c = np.ones(c1.shape, dtype=object)
        for i in range(c.shape[0]):
            c[i] = _add(params, c1[i], c2[i])
        return c
    # matrix + matrix
    elif isinstance(c1[0][0], int) and isinstance(c2[0][0], int) and c1.shape == c2.shape:
        c = np.ones(c1.shape, dtype=object)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                c[i][j] = _add(params, c1[i][j], c2[i][j])
        return c
    else:
        print('error: addtion')
        return None

def elementwise_add(params, c1, c2):
    if isinstance(c1, np.matrix):
        c1 = np.array(c1)
    if isinstance(c2, np.matrix):
        c2 = np.array(c2)

    # scalar + scalar
    if isinstance(c1, int) and isinstance(c2, int):
        return add(params, c1, c2)
    # vector + vector
    elif isinstance(c1[0], int) and isinstance(c2[0], int) and c1.shape == c2.shape:
        return add(params, c1, c2)
    # matrix + vector
    elif isinstance(c1[0][0], int) and isinstance(c2[0], int) and c1.shape[1] == c2.shape[0]:
        c = np.ones(c1.shape, dtype=object)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                c[i][j] = _add(params, c1[i][j], c2[j])
        return c
    # matrix + matrix
    elif isinstance(c1[0][0], int) and isinstance(c2[0][0], int) and c1.shape == c2.shape:
        return add(params, c1, c2)
    else:
        print('error: elementwise addtion')
        return None

def int_mult(params, m, c):
    if isinstance(m, np.matrix):
        m = np.array(m)
    if isinstance(c, np.matrix):
        c = np.array(c)

    # scalar (plaintext) x scalar (ciphertext)
    if isinstance(m, int) and isinstance(c, int):
        return _int_mult(params, m, c)
    # scalar (plaintext) x vector (ciphertext)
    elif isinstance(m, int) and isinstance(c[0], int):
        c_ = np.ones(c.shape, dtype=object)
        for i in range(c_.shape[0]):
            c_[i] = _int_mult(params, m, c[i])
        return c_
    # scalar (plaintext) x matrix (ciphertext)
    elif isinstance(m, int) and isinstance(c[0][0], int):
        c_ = np.ones(c.shape, dtype=object)
        for i in range(c_.shape[0]):
            for j in range(c_.shape[1]):
                c_[i][j] = _int_mult(params, m, c[i][j])
        return c_
    # vector (plaintext) x vector (ciphertext)
    elif isinstance(m[0], int) and isinstance(c[0], int) and m.shape == c.shape:
        c_ = 1
        for i in range(m.shape[0]):
            c_ = _add(params, c_, _int_mult(params, m[i], c[i]))
        return c_
    # matrix (plaintext) x vector (ciphertext)
    elif isinstance(m[0][0], int) and isinstance(c[0], int) and m.shape[1] == c.shape[0]:
        c_ = np.ones(m.shape[0], dtype=object)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                c_[i] = _add(params, c_[i], _int_mult(params, m[i][j], c[j]))
        return c_
    # matrix (plaintext) x matrix (ciphertext)
    elif isinstance(m[0][0], int) and isinstance(c[0][0], int) and m.shape[1] == c.shape[0]:
        c_ = np.ones([m.shape[0], c.shape[1]], dtype=object)
        for i in range(m.shape[0]):
            for j in range(c.shape[1]):
                for k in range(m.shape[1]):
                    c_[i][j] = _add(params, c_[i][j], _int_mult(params, m[i][k], c[k][j]))
        return c_
    else:
        print('error: integer multiplication')
        return None

def elementwise_int_mult(params, m, c):
    if isinstance(m, np.matrix):
        m = np.array(m)
    if isinstance(c, np.matrix):
        c = np.array(c)

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
    elif isinstance(m[0], int) and isinstance(c[0], int) and m.shape == c.shape:
        c_ = np.ones(c.shape, dtype=object)
        for i in range(c_.shape[0]):
            c_[i] = _int_mult(params, m[i], c[i])
        return c_
    # matrix (plaintext) x vector (ciphertext)
    elif isinstance(m[0][0], int) and isinstance(c[0], int) and m.shape[1] == c.shape[0]:
        c_ = np.ones(m.shape, dtype=object)
        for i in range(c_.shape[0]):
            for j in range(c_.shape[1]):
                c_[i][j] = _int_mult(params, m[i][j], c[j])
        return c_
    # matrix (plaintext) x matrix (ciphertext)
    elif isinstance(m[0][0], int) and isinstance(c[0][0], int) and m.shape == c.shape:
        c_ = np.ones(c.shape, dtype=object)
        for i in range(c_.shape[0]):
            for j in range(c_.shape[1]):
                c_[i][j] = _int_mult(params, m[i][j], c[i][j])
        return c_
    else:
        print('error: elementwise integer multiplication')
        return None

def encode(params, x, delta):
    f = np.frompyfunc(_encode, 3, 1)
    return f(params, x, delta)

def decode(params, m, delta):
    f = np.frompyfunc(_decode, 3, 1)
    return f(params, m, delta)

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
