import numpy as np

from eclib import regev

A = [
    [1.1, 2.2],
    [-3.3, 4.4],
]
x = [5.5, 6.6]
y = np.dot(A, x)
print(y)

sec_params = (10, 2**32, 2**64, 3.2)
params, pk, sk = regev.keygen(*sec_params)

s = 0.01
A_ecd = regev.encode(params, A, s)
x_enc = regev.enc(params, pk, x, s)
y_enc = regev.int_mult(params, A_ecd, x_enc)

y_ = regev.dec(params, sk, y_enc, s**2)
print(y_)
