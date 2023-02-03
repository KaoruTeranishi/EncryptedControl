#! /usr/bin/env python

# from eclib import regev, gsw
import regev, gsw
from eclib.numutils import *
from eclib.randutils import *
from eclib.primeutils import *
from eclib.modutils import *
from collections import namedtuple
from math import ceil, log2

def keygen(n, t, q, sigma, m=None):
    params = namedtuple('Parameters', ['n', 't', 'q', 'sigma', 'm', 'l', 'N'])

    if m == None:
        lwe_params, pk, sk = regev.keygen(n, t, q, sigma)
    else:
        lwe_params, pk, sk = regev.keygen(n, t, q, sigma, m)
    
    params.n = lwe_params.n
    params.t = lwe_params.t
    params.q = lwe_params.q
    params.sigma = lwe_params.sigma
    params.m = lwe_params.m
    params.l = ceil(log2(params.q))
    params.N = (params.n + 1) * params.l

    return params, pk, sk

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
