# Paillier encrytpion

This example illustrates how to compute a matrix-vector product using the Paillier encryption.

Import `numpy` package and `eclib.paillier` module.

```python
import numpy as np

from eclib import paillier
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

The key generation function `paillier.keygen()` requires to specify a key length for creating public and secret keys.
This example uses a key length of 128 bits.

```python
key_length = 128
params, pk, sk = paillier.keygen(key_length)
```

The matrix `A` and vector `x` are encoded and encrypted to `A_ecd` and `x_enc`, respectively, and `y_enc` is computed.

```python
s = 0.01
A_ecd = paillier.encode(params, A, s)
x_enc = paillier.enc(params, pk, x, s)
y_enc = paillier.int_mult(params, A_ecd, x_enc)
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
Similar to the [ElGamal encryption](./elgamal.md), the computation result can be recovered by the `paillier.dec()` function with $s^2$.

```python
y_ = paillier.dec(params, sk, y_enc, s**2)
print(y_)
```


## Code

```python
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
```
