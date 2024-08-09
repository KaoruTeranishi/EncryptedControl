# Dynamic-key ElGamal encryption

The [dynamic-key ElGamal encryption](https://ieeexplore.ieee.org/document/9774019) is a variant of the [ElGamal encryption](./elgamal.md) that allows for updating keys and ciphertexts.
This is useful for applications where the key pair needs to be updated periodically to enhance security.
ECLib implements the following algorithms.

## Key generation

The key generation algorithm is the same as the ElGamal encryption.


## Encryption

The encryption algorithm is the same as the ElGamal encryption.


## Decryption

The decryption algorithm is the same as the ElGamal encryption.


## Multiplication

The multiplication algorithm is the same as the ElGamal encryption.


## Key update

The key updating algorithm takes public parameters $(q, p, g)$, a public key $h$, and a secret key $s$ as input and outputs an updated public key $h'$, updated secret key $s'$, and token $t$,

$$
h' &= h g^r \bmod p, \\
s' &= s + r \bmod q, \\
t  &= (r, h),
$$

where $r \in \mathbb{Z}_q$ is a random number.


## Ciphertext update

The ciphertext update algorithm takes the public parameters $(q, p, g)$, a ciphertext $c = (c_1, c_2)$, and the token $t = (r, h)$ and outputs an updated ciphertext $c'$,

$$
c' = (c_1 g^u \bmod p, (c_1 g^u)^r c_2 h^u \bmod p),
$$

where $u \in \mathbb{Z}_q$ is a random number.
