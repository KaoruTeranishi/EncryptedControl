import numpy as np

from eclib import elgamal

A = [
    [1.1, 2.2],
    [-3.3, 4.4],
]
x = [5.5, 6.6]
y = np.dot(A, x)
print(y)

key_length = 128
params, pk, sk = elgamal.keygen(key_length)

s = 0.01
A_enc = elgamal.enc(params, pk, A, s)
x_enc = elgamal.enc(params, pk, x, s)
y_enc = elgamal.mult(params, A_enc, x_enc)

y_ = np.sum(elgamal.dec(params, sk, y_enc, s**2), axis=1)
print(y_)

y_ = elgamal.dec_add(params, sk, y_enc, s**2)
print(y_)
