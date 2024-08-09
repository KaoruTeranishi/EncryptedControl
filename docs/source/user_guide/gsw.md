# GSW encryption

The [GSW encryption](https://doi.org/10.1007/978-3-642-40041-4_5) is a public-key encryption scheme based on the Learning With Errors (LWE) problem.
ECLib implements the following algorithms.


## Key generation

The key generation algorithm takes $(m, n, q, \sigma)$ as input and outputs public parameters $(m, n, q, \sigma, \ell, N)$, public key $B$, and secret key $s$, where $m$, $n$, and $q$ are positive integers, $\sigma$ is a positive real number, $\ell = \lceil \log_2 q \rceil$, $N = (n + 1) \ell$, $s \in \mathbb{Z}_q^n$ is a random vector, $B = [b^\top \ A^\top]^\top \in \mathbb{Z}^{(n + 1) \times m}$, $A \in \mathbb{Z}_q^{n \times m}$ is a random matrix, $b = s^\top A + e^\top \bmod q$, and $e \in \mathbb{Z}^m$ is a random vector sampled from $m$-dimensional discrete Gaussian distribution with mean zero and variance $\sigma$.
The plaintext and ciphertext spaces are given by $\mathcal{M} = \mathbb{Z}_q$ and $\mathcal{C} = \mathbb{Z}_q^{(n + 1) \times N}$, respectively.


## Encryption

The encryption algorithm takes the public parameters, public key, and a plaintext $m \in \mathcal{M}$ as input and outputs

$$
B R + m G \bmod q,
$$

where $R \in \mathbb{Z}_2^{m \times N}$ is a random matrix, $G = I_{n + 1} \otimes g$ is a gadget matrix, $I_{n + 1}$ is the identify matrix of size $n + 1$, $g = [2^0 \ 2^1\ \cdots \ 2^{\ell - 1}]$, and $\otimes$ denotes the Kronecker product.


## Decryption

The decryption algorithm takes the public parameters, secret key, and a ciphertext $c \in \mathcal{C}$ as input.
Suppose that

$$
c &=
\begin{bmatrix}
    c_1 & \cdots & c_N
\end{bmatrix}, \quad
c_i \in \mathbb{Z}_q^{n + 1}, \\
\begin{bmatrix}
    x_0 \\
    \vdots \\
    x_{\ell - 1}
\end{bmatrix}
&=
\begin{bmatrix}
    1 & -s^\top
\end{bmatrix}
\begin{bmatrix}
    c_1 & \cdots & c_\ell
\end{bmatrix}
\bmod q.
$$

The algorithm outputs

$$
    \sum_{i=0}^{\ell-1} 2^i b_i \bmod q,
$$

where

$$
b_0 &= \left\lfloor \frac{ x_{\ell - 1} }{ 2^{\ell - 1} } \right\rceil \bmod 2, \\
b_i &= \left\lfloor \frac{ x_{\ell - i - 1} - \sum_{j=1}^i 2^{\ell - i + j - 2} x_{j - 1} }{ 2^{\ell - 1} } \right\rceil \bmod 2, \quad i = 1, \dots, \ell - 1.
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


## Multiplication

The multiplication algorithm takes the public parameters and two ciphertexts $c_1, c_2 \in \mathcal{C}$ as input and outputs

$$
c_1 
\begin{bmatrix}
    G^{-1}(c_{2,1}) & \cdots & G^{-1}(c_{2,N})
\end{bmatrix}
\bmod q,
$$

where $c_2 = [c_{2,1} \ \cdots \ c_{2,N}]$, and $G^{-1}: \mathbb{Z}_q^{n + 1} \to \{0, 1\}^N$ is a bit decomposition function such that $G G^{-1}(v) = v$ for all $v \in \mathbb{Z}_q^{n + 1}$.
