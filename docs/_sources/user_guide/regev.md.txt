# Regev (LWE) encryption

The [Regev encryption](https://doi.org/10.1145/1568318.1568324) is an encryption scheme based on the Learning With Errors (LWE) problem.
This encryption scheme has both public-key and secret-key encryption forms.
ECLib implements the public-key encryption algorithms.


## Key generation

The key generation algorithm takes $(m, n, t, q, \sigma)$ as input and outputs public parameters $(m, n, t, q, \sigma)$, public key $B$, and secret key $s$, where $m$, $n$, $t \ge 2$, and $q \gg t$ are positive integers, $\sigma$ is a positive real number, $s \in \mathbb{Z}_q^n$ is a random vector, $B = [b^\top \ A^\top]^\top \in \mathbb{Z}^{(n + 1) \times m}$, $A \in \mathbb{Z}_q^{n \times m}$ is a random matrix, $b = s^\top A + e^\top \bmod q$, and $e \in \mathbb{Z}^m$ is a random vector sampled from $m$-dimensional discrete Gaussian distribution with mean zero and variance $\sigma$.
The plaintext and ciphertext spaces are given by $\mathcal{M} = \mathbb{Z}_t$ and $\mathcal{C} = \mathbb{Z}_q^{n + 1}$, respectively.


## Encryption

The encryption algorithm takes the public parameters, public key, and a plaintext $m \in \mathcal{M}$ as input and outputs

$$
B r + m
\begin{bmatrix}
    \Delta \\
    \mathbf{0}_n
\end{bmatrix}
\bmod q,
$$

where $\Delta = \lfloor q / t \rfloor$, $r \in \mathbb{Z}_2^m$ is a random vector, and $\mathbf{0}_n$ is the $n$-dimensional zero vector.


## Decryption

The decryption algorithm takes the public parameters, secret key, and a ciphertext $c \in \mathcal{C}$ as input and outputs

$$
\left\lfloor \frac{t}{q} \left(
\begin{bmatrix}
    1 & -s^\top
\end{bmatrix}
c \bmod q\right)
\right\rceil \bmod t.
$$


## Addition

The addition algorithm takes the public parameters and two ciphertexts $c_1, c_2 \in \mathcal{C}$ as input and outputs

$$
c_1 + c_2 \bmod q.
$$


## Integer multiplication

The integer multiplication algorithm takes the public parameters, a plaintext $m \in \mathcal{M}$, and a ciphertext $c \in \mathcal{C}$ as input and outputs

$$
m c \bmod q.
$$
