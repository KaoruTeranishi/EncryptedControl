import numpy as np

from eclib import gsw_lwe

A = [
    [1.1, 2.2],
    [-3.3, 4.4],
]
x = [5.5, 6.6]
y = np.dot(A, x)
print(y)

sec_params = (10, 2**32, 2**64, 3.2)
params, pk, sk = gsw_lwe.keygen(*sec_params)

s = 0.01
A_enc = gsw_lwe.enc_gsw(params, pk, A, s)
x_enc = gsw_lwe.enc(params, pk, x, s)
y_enc = gsw_lwe.mult(params, A_enc, x_enc)

y_ = gsw_lwe.dec(params, sk, y_enc, s**2)
print(y_)
