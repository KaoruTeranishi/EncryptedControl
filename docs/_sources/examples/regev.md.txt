# Regev (LWE) encryption

This example illustrates how to compute a matrix-vector product using the Regev (LWE) encryption.

Import `numpy` package and `eclib.regev` module.

```python
import numpy as np

from eclib import regev
```

Define a matrix $A$ and a vector $x$ as

$$
A =
\begin{bmatrix}
     1.1 & 2.2 \\
    -3.3 & 4.4
\end{bmatrix}, \quad
x =
\begin{bmatrix}
    -5.5 \\
     6.6
\end{bmatrix},
$$

and compute $y = Ax$.

```python
A = [
    [1.1, 2.2],
    [-3.3, 4.4],
]
x = [-5.5, 6.6]
y = np.dot(A, x)
print(y)
```

The key generation function `regev.keygen()` requires to specify `m`, `n`, `t`, `q`, and `sigma` for creating public and secret keys, where `n` is the dimension of a lattice, which equals to the dimension of secret key, `m` is the subdimension of the lattice, `t` is the modulus of a plaintext space, `q` is the modulus of a ciphertext space, and `sigma` is the standard deviation of the discrete Gaussian distribution with mean zero used as an error distribution.
The parameter `m` is optional and is set to `2 * n * ceil(log2(q))` if not given.
This example omits `m` and uses `n = 10`, `t = 2**32`, `q = 2**64`, and `sigma = 3.2`.

```python
sec_params = (10, 2**32, 2**64, 3.2)
params, pk, sk = regev.keygen(*sec_params)
```

The matrix `A` and vector `x` are encoded and encrypted to `A_ecd` and `x_enc`, respectively, and `y_enc` is computed.

```python
s = 0.01
A_ecd = regev.encode(params, A, s)
x_enc = regev.enc(params, pk, x, s)
y_enc = regev.int_mult(params, A_ecd, x_enc)
```

Note that `A_ecd`, `x_enc`, and `y_enc` have the form

$$
A_\mathrm{ecd} &=
\begin{bmatrix}
    \bar{A}_{11} & \bar{A}_{12} \\
    \bar{A}_{21} & \bar{A}_{22}
\end{bmatrix}, \\
x_\mathrm{enc} &=
\begin{bmatrix}
    \mathsf{encrypt}(\bar{x}_1) \\
    \mathsf{encrypt}(\bar{x}_2)
\end{bmatrix}, \\
y_\mathrm{enc} &=
\begin{bmatrix}
    \mathsf{encrypt}(\bar{A}_{11} \bar{x}_1 + \bar{A}_{12} \bar{x}_2) \\
    \mathsf{encrypt}(\bar{A}_{21} \bar{x}_1 + \bar{A}_{22} \bar{x}_2)
\end{bmatrix},
$$

where $\bar{A}_{ij} = \mathsf{encode}(A_{ij} / s)$ and $\bar{x}_j = \mathsf{encode}(x_j / s)$.
Similar to the [ElGamal encryption](./elgamal.md), the computation result can be recovered by the `regev.dec()` function with $s^2$.

```python
y_ = regev.dec(params, sk, y_enc, s**2)
print(y_)
```


## Code

```python
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
```
