# Encoding and decoding

Control systems generally operate on real numbers, while encryption schemes work only on their plaintext spaces.
To encrypt control systems, it is therefore necessary to convert a floating-point number into plaintext.
ECLib provides encoder and decoder functions for this purpose.


## Encoding

Encoding function computes a plaintext from a floating-point number as follows.

```python
>>> s = 0.01
>>> x1 = 1.23
>>> m1 = encode(params, x1, s)
```

The floating-point number `x1` is scaled up to `x1 / s` and encoded to the nearest element in a plaintext space.
The specific process of `encode()` is different for each used encryption schemes.
The encoding function also supports negative numbers and 1D and 2D array-like inputs.

```python
>>> x2 = -4.56
>>> x3 = [1.23, -4.56, -7.89]
>>> m2 = encode(params, x2, s)
>>> m3 = encode(params, x3, s)
```

The scaling parameter `s` adjusts quantization errors due to the encoding process.
In most cases, the smaller `s`, the smaller the error.
Note that, however, the value of `s` is constrained by the size of plaintext space because it is a finite set.

The encryption schemes in ECLib provide encoding and encryption function `enc()` to simplify encrypting floating-point numbers.

```python
>>> c1 = enc(params, pk, x1, s)
>>> c2 = enc(params, pk, x2, s)
>>> c3 = enc(params, pk, x3, s)
```


## Decoding

Decoding function retrieves the floating-point number from the plaintext as follows.

```python
>>> x1_ = decode(params, m1, s)
```

It also supports 1D and 2D array-like plaintexts.

```python
>>> x2_ = decode(params, m2, s)
>>> x3_ = decode(params, m3, s)
```

Note that the decoded values `x1_`, `x2_`, and `x3_` are not necessarily the same as `x1`, `x2`, and `x3`, respectively, due to the quantization errors.
Similar to encoding function, decryption and decoding function `dec()` is availlable.

```python
>>> x1_ = dec(params, sk, c1, s)
>>> x2_ = dec(params, sk, c2, s)
>>> x3_ = dec(params, sk, c3, s)
```
