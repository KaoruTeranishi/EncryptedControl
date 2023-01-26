#! /usr/bin/env python

import eclib.elgamal as elgamal
from eclib.randutils import *
from eclib.modutils import *
from collections import namedtuple

def keygen(bit_length):
    return elgamal.keygen(bit_length)

def encrypt(params, pk, m):
    return elgamal.encrypt(params, pk, m)

def decrypt(params, sk, c):
    return elgamal.decrypt(params, sk, c)

def mult(params, c1, c2):
    return elgamal.mult(params, c1, c2)

def encode(params, x, delta, mode='nearest'):
    return elgamal.encode(params, x, delta, mode)

def decode(params, m, delta):
    return elgamal.decode(params, m, delta)

def enc(params, pk, x, delta, mode='nearest'):
    return elgamal.enc(params, pk, x, delta, mode)

def dec(params, sk, c, delta):
    return elgamal.dec(params, sk, c, delta)

def dec_add(params, sk, c, delta):
    return elgamal.dec_add(params, sk, c, delta)

def update_key(params, pk, sk):
    t = namedtuple('token', ['s', 'h'])
    t.s = get_rand(1, params.q)
    t.h = pk
    return (pk * mpow(params.g, t.s, params.p)) % params.p, (sk + t.s) % params.q, t

def update_ctxt(params, c, token):
    # scalar
    if isinstance(c[0], int):
        return _update_ctxt(params, c, token)
    # vector
    elif isinstance(c[0][0], int):
        c_ = [[0, 0] for i in range(len(c))]
        for i in range(len(c)):
            c_[i] = _update_ctxt(params, c[i], token)
        return c_
    # matrix
    elif isinstance(c[0][0][0], int):
        c_ = [[[0, 0] for j in range(len(c[0]))] for i in range(len(c))]
        for i in range(len(c)):
            for j in range(len(c[0])):
                c_[i][j] = _update_ctxt(params, c[i][j], token)
        return c_
    else:
        print('error: update ciphertext')
        return None

def _update_ctxt(params, c, t):
    r = get_rand(1, params.q)
    tmp = (c[0] * mpow(params.g, r, params.p)) % params.p
    return [tmp, (mpow(tmp, t.s, params.p) * c[1] * mpow(t.h, r, params.p)) % params.p]
