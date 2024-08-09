# Paillier encryption

The [Paillier encryption](https://doi.org/10.1007/3-540-48910-X_16) is a public-key encryption scheme based on the decisional composite residuosity assumption.
ECLib implements the following algorithms.


## Key generation


The key generation algorithm takes a key length $k$ as input and outputs public parameters $n$, public key $g$, and secret key $(\lambda, \mu)$, where $p$ and $q$ are $k$-bit prime numbers such that $\mathrm{gcd}(pq, (p - 1) (q - 1)) = 1$, $n = p q$, $\lambda = (p - 1) (q - 1)$, $\mu = \lambda^{-1} \bmod n$, and $g = n + 1$.
The plaintext and ciphertext spaces are $\mathcal{M} = \mathbb{Z}_n$ and $\mathcal{C} = \mathbb{Z}_n^\ast$, respectively.


## Encryption

The encryption algorithm takes the public parameters, public key, and a plaintext $m \in \mathcal{M}$ as input and outputs

$$
g^m r^n \bmod n^2,
$$

where $r \in \mathbb{Z}_n^\ast$ is a random number such that $\mathrm{gcd}(r, n) = 1$.


## Decryption

The decryption algorithm takes the public parameters, secret key, and a ciphertext $c \in \mathcal{C}$ as input and outputs

$$
\frac{ (c^\lambda \bmod n^2) - 1 }{ n } \mu \bmod n.
$$


## Addition

The addition algorithm takes the public parameters and two ciphertexts $c_1, c_2 \in \mathcal{C}$ as input and outputs

$$
c_1 c_2 \bmod n^2.
$$


## Integer multiplication

The integer multiplication algorithm takes the public parameters, a plaintext $m \in \mathcal{M}$, and a ciphertext $c \in \mathcal{C}$ as input and outputs

$$
c^m \bmod n^2.
$$
