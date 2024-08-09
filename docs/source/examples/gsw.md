# GSW encryption

This example illustrates how to compute a matrix-vector product using the GSW encryption.

Import `numpy` package and `eclib.gsw` module.

```python
import numpy as np

from eclib import gsw
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

The key generation function `gsw.keygen()` requires to specify `m`, `n`, `q`, and `sigma` for creating public and secret keys, where `n` is the dimension of a lattice, which equals to the dimension of secret key, `m` is the subdimension of the lattice, `q` is the modulus of plaintext and ciphertext spaces, and `sigma` is the standard deviation of the discrete Gaussian distribution with mean zero used as an error distribution.
The parameter `m` is optional and is set to `2 * n * ceil(log2(q))` if not given.
This example omits `m` and uses `n = 10`, `q = 2**64`, and `sigma = 3.2`.

```python
sec_params = (10, 2**64, 3.2)
params, pk, sk = gsw.keygen(*sec_params)
```

The matrix `A` and vector `x` are  encrypted to `A_enc` and `x_enc`, respectively, and `y_enc` is computed.

```python
s = 0.01
A_enc = gsw.enc(params, pk, A, s)
x_enc = gsw.enc(params, pk, x, s)
y_enc = gsw.mult(params, A_enc, x_enc)
```

Note that `A_enc`, `x_enc`, and `y_enc` have the form

$$
A_\mathrm{enc} &=
\begin{bmatrix}
    \mathsf{encrypt}(\bar{A}_{11}) & \mathsf{encrypt}(\bar{A}_{12}) \\
    \mathsf{encrypt}(\bar{A}_{21}) & \mathsf{encrypt}(\bar{A}_{22})
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
Similar to the [ElGamal encryption](./elgamal.md), the computation result can be recovered by the `gsw.dec()` function with $s^2$.

```python
y_ = gsw.dec(params, sk, y_enc, s**2)
print(y_)
```


## Code

```python
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
```
