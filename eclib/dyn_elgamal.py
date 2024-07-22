#! /usr/bin/env python3

from collections import namedtuple

import numpy as np

import eclib.modutils as mu
import eclib.randutils as ru
from eclib import elgamal


def keygen(bit_length):
    return elgamal.keygen(bit_length)


def encrypt(params, pk, m):
    return elgamal.encrypt(params, pk, m)


def decrypt(params, sk, c):
    return elgamal.decrypt(params, sk, c)


def mult(params, c1, c2):
    return elgamal.mult(params, c1, c2)


def encode(params, x, delta, mode="nearest"):
    return elgamal.encode(params, x, delta, mode)


def decode(params, m, delta):
    return elgamal.decode(params, m, delta)


def enc(params, pk, x, delta, mode="nearest"):
    return elgamal.enc(params, pk, x, delta, mode)


def dec(params, sk, c, delta):
    return elgamal.dec(params, sk, c, delta)


def dec_add(params, sk, c, delta):
    return elgamal.dec_add(params, sk, c, delta)


def update_key(params, pk, sk):
    t = namedtuple("token", ["s", "h"])
    t.s = ru.get_rand(1, params.q)
    t.h = pk
    return (pk * mu.mpow(params.g, t.s, params.p)) % params.p, (sk + t.s) % params.q, t


def update_ct(params, c, token):
    # scalar
    if isinstance(c[0], int):
        return _update_ct(params, c, token)
    # vector
    elif isinstance(c[0][0], int):
        c_ = np.zeros(c.shape, dtype=object)
        for i in range(c_.shape[0]):
            c_[i] = _update_ct(params, c[i], token)
        return c_
    # matrix
    elif isinstance(c[0][0][0], int):
        c_ = np.zeros(c.shape, dtype=object)
        for i in range(c_.shape[0]):
            for j in range(c_.shape[1]):
                c_[i][j] = _update_ct(params, c[i][j], token)
        return c_
    else:
        print("error: update ciphertext")
        return None


def _update_ct(params, c, t):
    r = ru.get_rand(1, params.q)
    tmp = (c[0] * mu.mpow(params.g, r, params.p)) % params.p
    return np.array(
        [
            tmp,
            (mu.mpow(tmp, t.s, params.p) * c[1] * mu.mpow(t.h, r, params.p)) % params.p,
        ],
        dtype=object,
    )
