# GSW-LWE encryption

The outer product of [GSW](./gsw.md) and [LWE](./regev.md) ciphertexts enables to compute another LWE ciphertext corresponding to their multiplication.
This is useful for efficient multiplication by suppressing error growth in LWE encryption.
The GSW-LWE encryption in ECLib implements the following algorithms to provide the functionality.

## Key generation

The key generation algorithm takes $(m, n, t, q, \sigma)$ as input and outputs public parameters, a public key, and a secret key.
The public parameters consists of GSW and LWE public parameters, and the public and secret keys are the same as the LWE encryption schemes.


## Encryption

The GSW-LWE encryption scheme provides both the GSW and LWE encryption algorithms.


## Decryption

The GSW-LWE encryption scheme provides both the GSW and LWE decryption algorithms.


## Addition

The addition algorithm is the same as the LWE encryption scheme.


## Integer multiplication

The integer multiplication algorithm is the same as the LWE encryption scheme.


## Multiplication

The multiplication algorithm takes the public parameters and a GSW ciphertext $c_\mathrm{GSW}$ and a LWE ciphertext $c_\mathrm{LWE}$ as input and outputs

$$
c_\mathrm{GSW} \cdot G^{-1}(c_\mathrm{LWE}) \bmod q,
$$

where $G^{-1}$ is the bit decomposition function used in the [GSW](./gsw.md) encryption.

