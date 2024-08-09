import numpy as np

from eclib import paillier

A = [
    [1.1, 2.2],
    [-3.3, 4.4],
]
x = [5.5, 6.6]
y = np.dot(A, x)
print(y)

key_length = 128
params, pk, sk = paillier.keygen(key_length)

s = 0.01
A_ecd = paillier.encode(params, A, s)
x_enc = paillier.enc(params, pk, x, s)
y_enc = paillier.int_mult(params, A_ecd, x_enc)

y_ = paillier.dec(params, sk, y_enc, s**2)
print(y_)
