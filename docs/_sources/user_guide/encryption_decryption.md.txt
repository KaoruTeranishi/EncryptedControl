# Encryption and decryption


## Encryption

Encryption function computes a ciphertext from a plaintext as follows.

```python
>>> m = 1
>>> c = encrypt(params, pk, m)
```

The plaintext `m` is encrypted by the function using the public key `pk`.
The ciphertext `c` behaves as a random variable over a ciphertext space, and its value is different for each run.
Note that a message must be in a plaintext space to be encrypted.
If not, the encrypted message will not necessarily be decrypted correctly.
The encryption function allows 1D and 2D array-like plaintexts as input.
Even in such a case, it works in the same syntax.

```python
>>> m_v = [1, 2, 3]
>>> c_v = encrypt(params, pk, m_v)
```


## Decryption

Decryption function recovers the plaintext from the ciphertext as follows.

```python
>>> m_ = decrypt(params, sk, c)
>>> print(m_)
1
```

The ciphertext `c` is decrypted to the original message by the function using the secret key `sk`.
The decryption process is deterministic unlike encryption.
Similar to the encryption function, the decryption function supports 1D and 2D array-like ciphertexts. 

```python
>>> m_v_ = decrypt(params, sk, c_v)
>>> print(m_v_)
[1 2 3]
```