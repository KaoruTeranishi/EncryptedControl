# ElGamal encryption

This example illustrates how to compute a matrix-vector product using the ElGamal encryption.

To begin with, import `numpy` package and `eclib.elgamal` module.

```python
import numpy as np

from eclib import elgamal
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

The key generation function `elgamal.keygen()` requires to specify a key length for creating public and secret keys.
This example uses a key length of 128 bits.

```python
key_length = 128
params, pk, sk = elgamal.keygen(key_length)
```

The matrix `A` and vector `x` are encrypted to `A_enc` and `x_enc` using the public key `pk`, respectively, and `y_enc` is computed.

```python
s = 0.01
A_enc = elgamal.enc(params, pk, A, s)
x_enc = elgamal.enc(params, pk, x, s)
y_enc = elgamal.mult(params, A_enc, x_enc)
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
    \mathsf{encrypt}(\bar{A}_{11} \bar{x}_1) & \mathsf{encrypt}(\bar{A}_{12} \bar{x}_2) \\
    \mathsf{encrypt}(\bar{A}_{21} \bar{x}_1) & \mathsf{encrypt}(\bar{A}_{22} \bar{x}_2)
\end{bmatrix},
$$

where $\bar{A}_{ij} = \mathsf{encode}(A_{ij} / s)$ and $\bar{x}_j = \mathsf{encode}(x_j / s)$.
The `elgamal.enc()` function internally calls the `elgamal.encode()` and `elgamal.encrypt()` functions.
Each element of `A` and `x` is converted to the nearest plaintext after scaling to `A / s` and `x / s` by the `elgamal.encode()` function, and then encrypted by the `elgamal.encrypt()` function.
The multiplication function `elgamal.mult()` returns the Hadamard product for 1D or 2D array-like inputs.
Thus, `y_enc` become a matrix given by the encryption of the element-wise product between `A / s` and `x / s`.

To recover the computation result, each element of `y_enc` is decrytped using the secret key `sk` and decoded with $s^2$ by the `elgamal.dec()` function, and then summed up for each row.

```python
y_ = np.sum(elgamal.dec(params, sk, y_enc, s**2), axis=1)
print(y_)
```

Alternatively, you can use the `elgamal.dec_add()` function.

```python
y_ = elgamal.dec_add(params, sk, y_enc, s**2)
print(y_)
```


## Code

```python
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
```
