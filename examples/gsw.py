import numpy as np

from eclib import gsw

A = [
    [1.1, 2.2],
    [-3.3, 4.4],
]
x = [5.5, 6.6]
y = np.dot(A, x)
print(y)

sec_params = (10, 2**64, 3.2)
params, pk, sk = gsw.keygen(*sec_params)

s = 0.01
A_enc = gsw.enc(params, pk, A, s)
x_enc = gsw.enc(params, pk, x, s)
y_enc = gsw.mult(params, A_enc, x_enc)

y_ = gsw.dec(params, sk, y_enc, s**2)
print(y_)
