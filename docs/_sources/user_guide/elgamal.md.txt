# ElGamal encryption

The [ElGamal encryption](https://doi.org/10.1109/TIT.1985.1057074) is a public-key encryption scheme based on the Decisional Diffie-Hellman (DDH) assumption.
ECLib implements the following algorithms.


## Key generation

The key generation algorithm takes a key length $k$ as input and outputs public parameters $(q, p, g)$, public key $h$, and secret key $s$, where $q$ is a $k$-bit Sophie Germain prime, $p = 2q + 1$ is a corresponding safe prime, $g$ is a generator of a cyclic group $\mathbb{G} = \{ g^i \bmod p \mid i \in \mathbb{Z}_q \}$ such that $g^q = 1 \bmod p$, $s \in \mathbb{Z}_q$ is a random number, and $h = g^s \bmod p$.
The plaintext and ciphertext spaces are given by $\mathcal{M} = \mathbb{G}$ and $\mathcal{C} = \mathbb{G}^2$, respectively.


## Encryption

The encryption algorithm takes the public parameters, public key, and a plaintext $m \in \mathcal{M}$ as input and outputs

$$
(g^r \bmod p, m h^r \bmod p),
$$

where $r \in \mathbb{Z}_q$ is a random number.


## Decryption

The decryption algorithm takes the public parameters, secret key, and a ciphertext $c = (c_1, c_2) \in \mathcal{C}$ as input and outputs

$$
c_1 {c_2}^{-s} \bmod p.
$$


## Multiplication

The multiplication algorithm takes the public parameters and two ciphertexts $c_1 = (c_{11}, c_{12}), c_2 = (c_{21}, c_{22}) \in \mathcal{C}$ as input and outputs

$$
(c_{11} c_{21} \bmod p, c_{12} c_{22} \bmod p).
$$
