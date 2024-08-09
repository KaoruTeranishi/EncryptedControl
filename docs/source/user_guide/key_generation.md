# Key generation

Key generation function creates public parameters, a public key, and a secret key as follows.

```python
>>> params, pk, sk = keygen(sec_params)
```

The argument `sec_params` is `int` or `tuple` to decide the security level of encryption scheme.
Its detailed value depends on the used encryption scheme (see [Examples](../examples/examples.rst)).


## Public parameters

Public parameters `params` consists of parameters that is published and specify the context of used encryption scheme, such as plaintext and ciphertext spaces.
All the functions in encryption scheme modules take this variable as input.


## Public key

Pbulic key `pk` is a key for encryption to be published.

## Secret key

Secret key `sk` is a key for decryption to be kept secret.
