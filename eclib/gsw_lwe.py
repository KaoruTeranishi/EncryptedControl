#! /usr/bin/env python

from eclib.numutils import *
from randutils import *
from eclib.primeutils import *
from eclib.modutils import *
from collections import namedtuple
import numpy as np
from math import floor, log2

def keygen(bit_length, n, sigma):
    params = namedtuple('Parameters', ['q', 'n', 'l', 'N', 'sigma'])
    params.q = get_prime(bit_length)
    params.n = n
    params.l = floor(log2(params.q)) + 1
    params.N = (params.n + 1) * params.l
    params.sigma = sigma

    sk = -np.array(get_int_gaussian(0, params.sigma, params.n), dtype=object).reshape(-1, 1) % params.q

    return params, sk

def encrypt(params, pk, m):
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

def encrypt_gsw(params, pk, m):
    # scalar
    if isinstance(m, int):
        return _encrypt_gsw(params, pk, m)
    # vector
    elif isinstance(m[0], int):
        c = np.zeros(m.shape, dtype=object)
        for i in range(c.shape[0]):
            c[i] = _encrypt_gsw(params, pk, m[i])
        return c
    # matrix
    elif isinstance(m[0][0], int):
        c = np.zeros(m.shape, dtype=object)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                c[i][j] = _encrypt_gsw(params, pk, m[i][j])
        return c
    else:
        print('error: encryption (GSW)')
        return None

def decrypt(params, sk, c):
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
    # scalar + scalar
    if isinstance(c1[0][0], int) and isinstance(c2[0][0], int):
        return add(params, c1, c2)
    # vector + vector
    elif isinstance(c1[0][0][0], int) and isinstance(c2[0][0][0], int) and c1.shape == c2.shape:
        return add(params, c1, c2)
    # matrix + vector
    elif isinstance(c1[0][0][0][0], int) and isinstance(c2[0][0][0], int) and c1.shape[1] == c2.shape[0]:
        c = np.zeros(c1.shape, dtype=object)
        for i in range(len(c)):
            for j in range(len(c[0])):
                c[i][j] = _add(params, c1[i][j], c2[j])
        return c
    # matrix + matrix
    elif isinstance(c1[0][0][0][0], int) and isinstance(c2[0][0][0][0], int) and c1.shape == c2.shape:
        return add(params, c1, c2)
    else:
        print('error: elementwise addtion')
        return None

def mult(params, c1, c2):
    # scalar (GSW-ciphertext) x scalar (ciphertext)
    if isinstance(c1[0][0], int) and isinstance(c2[0][0], int):
        return _mult(params, c1, c2)
    # scalar (GSW-ciphertext) x vector (ciphertext)
    elif isinstance(c1[0][0], int) and isinstance(c2[0][0][0], int):
        c_ = np.zeros(c2.shape, dtype=object)
        for i in range(c_.shape[0]):
            c_[i] = _mult(params, c1, c2[i])
        return c_
    # scalar (GSW-ciphertext) x matrix (ciphertext)
    elif isinstance(c1[0][0], int) and isinstance(c2[0][0][0][0], int):
        c_ = np.zeros(c2.shape, dtype=object)
        for i in range(c_.shape[0]):
            for j in range(c_.shape[1]):
                c_[i][j] = _mult(params, c1, c2[i][j])
        return c_
    # vector (GSW-ciphertext) x vector (ciphertext)
    elif isinstance(c1[0][0][0], int) and isinstance(c2[0][0][0], int) and c1.shape == c2.shape:
        c_ = np.zeros(1, dtype=object)
        for i in range(c1.shape[0]):
            c_ = _add(params, c_, _mult(params, c1[i], c2[i]))
        return c_
    # matrix (GSW-ciphertext) x vector (ciphertext)
    elif isinstance(c1[0][0][0][0], int) and isinstance(c2[0][0][0], int) and c1.shape[1] == c2.shape[0]:
        c_ = np.zeros(c1.shape[0], dtype=object)
        for i in range(c1.shape[0]):
            for j in range(c1.shape[1]):
                c_[i] = _add(params, c_[i], _mult(params, c1[i][j], c2[j]))
        return c_
    # matrix (GSW-ciphertext) x matrix (ciphertext)
    elif isinstance(c1[0][0][0][0], int) and isinstance(c2[0][0][0][0], int) and c1.shape[1] == c2.shape[0]:
        c_ = np.zeros([c1.shape[0], c2.shape[1]], dtype=object)
        for i in range(c1.shape[0]):
            for j in range(c2.shape[1]):
                for k in range(c1.shape[1]):
                    c_[i][j] = _add(params, c_[i][j], _mult(params, c1[i][k], c2[k][j]))
        return c_
    else:
        print('error: multiplication')
        return None

def elementwise_mult(params, c1, c2):
    # scalar (GSW-ciphertext) x scalar (ciphertext)
    if isinstance(c1[0][0], int) and isinstance(c2[0][0], int):
        return mult(params, c1, c2)
    # scalar (GSW-ciphertext) x vector (ciphertext)
    elif isinstance(c1[0][0], int) and isinstance(c2[0][0][0], int):
        return mult(params, c1, c2)
    # scalar (GSW-ciphertext) x matrix (ciphertext)
    elif isinstance(c1[0][0], int) and isinstance(c2[0][0][0][0], int):
        return mult(params, c1, c2)
    # vector (GSW-ciphertext) x vector (ciphertext)
    elif isinstance(c1[0][0][0], int) and isinstance(c2[0][0][0], int) and c1.shape == c2.shape:
        c_ = np.zeros(c1.shape, dtype=object)
        for i in range(c_.shape[0]):
            c_[i] = _mult(params, c1[i], c2[i])
        return c_
    # matrix (GSW-ciphertext) x vector (ciphertext)
    elif isinstance(c1[0][0][0][0], int) and isinstance(c2[0][0][0], int) and c1.shape[1] == c2.shape[0]:
        c_ = np.zeros(c1.shape, dtype=object)
        for i in range(c_.shape[0]):
            for j in range(c_.shape[1]):
                c_[i][j] = _mult(params, c1[i][j], c2[j])
        return c_
    # matrix (GSW-ciphertext) x matrix (ciphertext)
    elif isinstance(c1[0][0][0][0], int) and isinstance(c2[0][0][0][0], int) and c1.shape == c2.shape:
        c_ = np.zeros(c1.shape, dtype=object)
        for i in range(c_.shape[0]):
            for j in range(c_.shape[1]):
                c_[i][j] = _mult(params, c1[i][j], c2[i][j])
        return c_
    else:
        print('error: elementwise multiplication')
        return None

def int_mult(params, m, c):
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

def enc_gsw(params, pk, x, delta):
    return encrypt_gsw(params, pk, encode(params, x, delta))

def dec(params, sk, c, delta):
    return decode(params, decrypt(params, sk, c), delta)

def _encrypt(params, sk, m):
    a = np.array([[get_rand(0, params.q)] for _ in range(params.n)], dtype=object)
    e = get_int_gaussian(0, params.sigma) % params.q
    return np.block([[(sk.T @ a + m + e) % params.q], [a]])

def _encrypt_gsw(params, sk, m):
    A = np.array([[get_rand(0, params.q) for _ in range(params.N)] for _ in range(params.n)], dtype=object)
    E = np.array(get_int_gaussian(0, params.sigma, params.N), dtype=object).reshape(-1, 1) % params.q
    G = _gadget(params)
    return (m * G + np.block([[(sk.T @ A + E.T) % params.q], [A]])) % params.q

def _decrypt(params, sk, c):
    return (np.block([1, -sk.T]) @ c)[0][0] % params.q

def _add(params, c1, c2):
    return (c1 + c2) % params.q

def _int_mult(params, m, c):
    return (m * c) % params.q

def _mult(params, c1, c2):
    return (c1 @ _bitdecomp(params, c2)) % params.q

def _encode(params, x, delta):
    m = floor(x / delta + 0.5)

    if m < 0:
        if m < -floor((params.q - 1) / 2):
            print('error: underflow')
            return None
        else:
            m += params.q
    elif m > floor(params.q / 2):
        print('error: overflow')
        return None

    return m

def _decode(params, m, delta):
    return min_residue(m, params.q) * delta

def _gadget(params):
    g = 2 ** np.arange(params.l, dtype=object)
    return np.kron(g, np.identity(params.n + 1, dtype=object))

def _bitdecomp(params, v):
    tmp = [np.binary_repr(x, params.l)[::-1] for x in v.T[0]]
    return np.array([[int(tmp[i][j]) for i in range(params.n + 1)] for j in range(params.l)]).reshape(-1, 1)
