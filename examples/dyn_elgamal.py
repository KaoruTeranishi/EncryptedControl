import numpy as np

from eclib import dyn_elgamal

A = [
    [1.1, 2.2],
    [-3.3, 4.4],
]
x = [5.5, 6.6]
y = np.dot(A, x)
print(y)

key_length = 128
params, pk, sk = dyn_elgamal.keygen(key_length)
print(pk)
print(sk)

s = 0.01
A_enc = dyn_elgamal.enc(params, pk, A, s)

pk, sk, t = dyn_elgamal.update_key(params, pk, sk)
x_enc = dyn_elgamal.enc(params, pk, x, s)
print(pk)
print(sk)

A_enc = dyn_elgamal.update_ct(params, A_enc, t)
y_enc = dyn_elgamal.mult(params, A_enc, x_enc)

y_ = dyn_elgamal.dec_add(params, sk, y_enc, s**2)
print(y_)
