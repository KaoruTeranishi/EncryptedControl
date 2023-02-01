#! /usr/bin/env python

from eclib import regev, gsw
from eclib.numutils import *
from eclib.randutils import *
from eclib.primeutils import *
from eclib.modutils import *

def keygen(n, q, sigma, m=None):
    if m == None:
        return gsw.keygen(n, q, sigma)
    else:
        return gsw.keygen(n, q, sigma, m)

def encrypt(params, pk, m):
    return regev.encrypt(params, pk, m)

def encrypt_gsw(params, pk, m):
    return gsw.encrypt(params, pk, m)

def decrypt(params, sk, c):
    return regev.decrypt(params, sk, c)

def add(params, c1, c2):
    return regev.add(params, c1, c2)

def elementwise_add(params, c1, c2):
    return regev.elementwise_add(params, c1, c2)

def mult(params, c1, c2):
    return gsw.mult(params, c1, c2)

def elementwise_mult(params, c1, c2):
    return gsw.elementwise_mult(params, c1, c2)

def int_mult(params, m, c):
    return regev.int_mult(params, m, c)

def elementwise_int_mult(params, m, c):
    return regev.elementwise_int_mult(params, m, c)

def encode(params, x, delta):
    return regev.encode(params, x, delta)

def decode(params, m, delta):
    return regev.decode(params, m, delta)

def enc(params, pk, x, delta):
    return regev.enc(params, pk, x, delta)

def enc_gsw(params, pk, x, delta):
    return gsw.enc(params, pk, x, delta)

def dec(params, sk, c, delta):
    return regev.dec(params, sk, c, delta)
