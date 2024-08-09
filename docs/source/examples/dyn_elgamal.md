# Dynamic-key ElGamal encryption

This example illustrates how to compute a matrix-vector product using the dynamic-key ElGamal encryption with key update.

Import `numpy` package and `eclib.dyn_elgamal` module.

```python
import numpy as np

from eclib import dyn_elgamal
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

The key generation function `elgamal.dyn_keygen()` requires to specify a key length for creating public and secret keys.
This example uses a key length of 128 bits.

```python
key_length = 128
params, pk, sk = dyn_elgamal.keygen(key_length)
print(pk)
print(sk)
```

The matrix `A` is encrypted to `A_enc` using the public key `pk`.

```python
s = 0.01
A_enc = dyn_elgamal.enc(params, pk, A, s)
```

The public and secret keys are updated, and then the vector `x` is encrypted to `x_enc` using the updated public key.

```python
pk, sk, t = dyn_elgamal.update_key(params, pk, sk)
x_enc = dyn_elgamal.enc(params, pk, x, s)
print(pk)
print(sk)
```

In addition, `A_enc` is udpated to correspond with the updated keys using the token `t`, and `y_enc` is computed.

```python
A_enc = dyn_elgamal.update_ct(params, A_enc, t)
y_enc = dyn_elgamal.mult(params, A_enc, x_enc)
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
Similar to the [ElGamal encryption](./elgamal.md), the computation result can be recovered by the `dyn_elgamal.dec_add()` function with $s^2$.

```python
y_ = dyn_elgamal.dec_add(params, sk, y_enc, s**2)
print(y_)
```


## Code

```python
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
```
