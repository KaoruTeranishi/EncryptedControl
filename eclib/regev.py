#! /usr/bin/env python

from eclib.numutils import *
from eclib.randutils import *
from eclib.primeutils import *
from eclib.modutils import *
from collections import namedtuple
import numpy as np
from math import floor, ceil, log2

def keygen(n, t, q, sigma, m=None):
    params = namedtuple('Parameters', ['n', 't', 'q', 'sigma', 'm'])
    params.n = n
    params.t = t
    params.q = q
    params.sigma = sigma
    if m == None:
        params.m = 2 * params.n * ceil(log2(params.q))
    else:
        params.m = m

    A = np.array([[get_rand(0, params.q) for _ in range(params.m)] for _ in range(params.n)], dtype=object)
    s = np.array([[get_rand(0, params.q)] for _ in range(params.n)], dtype=object)
    e = np.array(get_int_gaussian(0, sigma, params.m), dtype=object).reshape(-1, 1)

    pk = np.block([[(s.T @ A + e.T) % params.q], [A]])
    sk = np.block([1, -s.T])

    return params, pk, sk

def encrypt(params, pk, m):
    if isinstance(m, np.matrix) or isinstance(m, list):
        m = np.array(m)

    # scalar
    if isinstance(m, int):
        return _encrypt(params, pk, m)
    # vector
    elif isinstance(m[0], int):
        c = np.zeros(m.shape, dtype=object)
        for i in range(c.shape[0]):
            c[i] = _encrypt(params, pk, m[i])
        return c
    # matrix
    elif isinstance(m[0][0], int):
        c = np.zeros(m.shape, dtype=object)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                c[i][j] = _encrypt(params, pk, m[i][j])
        return c
    else:
        print('error: encryption')
        return None

def decrypt(params, sk, c):
    if isinstance(c, np.matrix) or isinstance(c, list):
        c = np.array(c)

    # scalar
    if isinstance(c[0][0], int):
        return _decrypt(params, sk, c)
    # vector
    elif isinstance(c[0][0][0], int):
        m = np.zeros(c.shape, dtype=object)
        for i in range(m.shape[0]):
            m[i] = _decrypt(params, sk, c[i])
        return m
    # matrix
    elif isinstance(c[0][0][0][0], int):
        m = np.zeros(c.shape, dtype=object)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                m[i][j] = _decrypt(params, sk, c[i][j])
        return m
    else:
        print('error: decryption')
        return None

def add(params, c1, c2):
    if isinstance(c1, np.matrix) or isinstance(c1, list):
        c1 = np.array(c1)
    if isinstance(c2, np.matrix) or isinstance(c2, list):
        c2 = np.array(c2)

    # scalar + scalar
    if isinstance(c1[0][0], int) and isinstance(c2[0][0], int):
        return _add(params, c1, c2)
    # vector + vector
    elif isinstance(c1[0][0][0], int) and isinstance(c2[0][0][0], int) and c1.shape == c2.shape:
        c = np.zeros(c1.shape, dtype=object)
        for i in range(c.shape[0]):
            c[i] = _add(params, c1[i], c2[i])
        return c
    # matrix + matrix
    elif isinstance(c1[0][0][0][0], int) and isinstance(c2[0][0][0][0], int) and c1.shape == c2.shape:
        c = np.zeros(c1.shape, dtype=object)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                c[i][j] = _add(params, c1[i][j], c2[i][j])
        return c
    else:
        print('error: addition')
        return None

def elementwise_add(params, c1, c2):
    if isinstance(c1, np.matrix) or isinstance(c1, list):
        c1 = np.array(c1)
    if isinstance(c2, np.matrix) or isinstance(c2, list):
        c2 = np.array(c2)

    # scalar + scalar
    if isinstance(c1[0][0], int) and isinstance(c2[0][0], int):
        return add(params, c1, c2)
    # vector + vector
    elif isinstance(c1[0][0][0], int) and isinstance(c2[0][0][0], int) and c1.shape == c2.shape:
        return add(params, c1, c2)
    # matrix + vector
    elif isinstance(c1[0][0][0][0], int) and isinstance(c2[0][0][0], int) and c1.shape[1] == c2.shape[0]:
        c = np.zeros(c1.shape, dtype=object)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                c[i][j] = _add(params, c1[i][j], c2[j])
        return c
    # matrix + matrix
    elif isinstance(c1[0][0][0][0], int) and isinstance(c2[0][0][0][0], int) and c1.shape == c2.shape:
        return add(params, c1, c2)
    else:
        print('error: elementwise addtion')
        return None

def int_mult(params, m, c):
    if isinstance(m, np.matrix) or isinstance(m, list):
        m = np.array(m)
    if isinstance(c, np.matrix) or isinstance(c, list):
        c = np.array(c)

    # scalar (plaintext) x scalar (ciphertext)
    if isinstance(m, int) and isinstance(c[0][0], int):
        return _int_mult(params, m, c)
    # scalar (plaintext) x vector (ciphertext)
    elif isinstance(m, int) and isinstance(c[0][0][0], int):
        c_ = np.zeros(c.shape, dtype=object)
        for i in range(c_.shape[0]):
            c_[i] = _int_mult(params, m, c[i])
        return c_
    # scalar (plaintext) x matrix (ciphertext)
    elif isinstance(m, int) and isinstance(c[0][0][0][0], int):
        c_ = np.zeros(c.shape, dtype=object)
        for i in range(c_.shape[0]):
            for j in range(c_.shape[1]):
                c_[i][j] = _int_mult(params, m, c[i][j])
        return c_
    # vector (plaintext) x vector (ciphertext)
    elif isinstance(m[0], int) and isinstance(c[0][0][0], int) and m.shape == c.shape:
        c_ = np.zeros(1, dtype=object)
        for i in range(m.shape[0]):
            c_ = _add(params, c_, _int_mult(params, m[i], c[i]))
        return c_
    # matrix (plaintext) x vector (ciphertext)
    elif isinstance(m[0][0], int) and isinstance(c[0][0][0], int) and m.shape[1] == c.shape[0]:
        c_ = np.zeros(m.shape[0], dtype=object)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                c_[i] = _add(params, c_[i], _int_mult(params, m[i][j], c[j]))
        return c_
    # matrix (plaintext) x matrix (ciphertext)
    elif isinstance(m[0][0], int) and isinstance(c[0][0][0][0], int) and m.shape[1] == c.shape[0]:
        c_ = np.zeros([m.shape[0], c.shape[1]], dtype=object)
        for i in range(m.shape[0]):
            for j in range(c.shape[1]):
                for k in range(m.shape[1]):
                    c_[i][j] = _add(params, c_[i][j], _int_mult(params, m[i][k], c[k][j]))
        return c_
    else:
        print('error: integer multiplication')
        return None

def elementwise_int_mult(params, m, c):
    if isinstance(m, np.matrix) or isinstance(m, list):
        m = np.array(m)
    if isinstance(c, np.matrix) or isinstance(c, list):
        c = np.array(c)

    # scalar (plaintext) x scalar (ciphertext)
    if isinstance(m, int) and isinstance(c[0][0], int):
        return int_mult(params, m, c)
    # scalar (plaintext) x vector (ciphertext)
    elif isinstance(m, int) and isinstance(c[0][0][0], int):
        return int_mult(params, m, c)
    # scalar (plaintext) x matrix (ciphertext)
    elif isinstance(m, int) and isinstance(c[0][0][0][0], int):
        return int_mult(params, m, c)
    # vector (plaintext) x vector (ciphertext)
    elif isinstance(m[0], int) and isinstance(c[0][0][0], int) and m.shape == c.shape:
        c_ = np.zeros(c.shape, dtype=object)
        for i in range(c_.shape[0]):
            c_[i] = _int_mult(params, m[i], c[i])
        return c_
    # matrix (plaintext) x vector (ciphertext)
    elif isinstance(m[0][0], int) and isinstance(c[0][0][0], int) and m.shape[1] == c.shape[0]:
        c_ = np.zeros(m.shape, dtype=object)
        for i in range(c_.shape[0]):
            for j in range(c_.shape[1]):
                c_[i][j] = _int_mult(params, m[i][j], c[j])
        return c_
    # matrix (plaintext) x matrix (ciphertext)
    elif isinstance(m[0][0], int) and isinstance(c[0][0][0][0], int) and m.shape == c.shape:
        c_ = np.zeros(c.shape, dtype=object)
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

def _encrypt(params, pk, m):
    r = np.array([[get_rand(0, 2)] for _ in range(params.m)], dtype=object)
    return (pk @ r + floor(params.q / params.t) * m * np.block([[1], [np.zeros([params.n, 1], dtype=object)]])) % params.q

def _decrypt(params, sk, c):
    return floor((params.t / params.q) * ((sk @ c)[0][0] % params.q) + 0.5) % params.t

def _add(params, c1, c2):
    return (c1 + c2) % params.q

def _int_mult(params, m, c):
    return (m * c) % params.q

def _encode(params, x, delta):
    m = floor(x / delta + 0.5)

    if m < 0:
        if m < -floor((params.t - 1) / 2):
            print('error: underflow')
            return None
        else:
            m += params.t
    elif m > floor(params.t / 2):
        print('error: overflow')
        return None

    return m

def _decode(params, m, delta):
    return min_residue(m, params.t) * delta
