# Homomorphic operation

Homomorphic encryption schemes in ECLib are categorized into multiplicatively, additively and fully homomorphic encryption.
Each class of encryption scheme supports different types of homomorphic operations.


## Multiplicatively homomorphic encryption

Multiplicatively homomorphic encryption supports multiplication as a homomorphic operation.

```python
>>> c1 = encrypt(params, pk, m1)
>>> c2 = encrypt(params, pk, m2)
>>> c3 = mult(params, c1, c2)
```

The decryption of `c3` is expected to be `m1 * m2`.
It can also compute the Hadamard product for 1D and 2D array-like ciphertexts.

```python
>>> c_v1 = encrypt(params, pk, [m1, m2, m3])
>>> c_v2 = encrypt(params, pk, [m4, m5, m6])
>>> c_v3 = mult(params, c_v1, c_v2)
```

The decryption of `c_v3` is expected to be the array that consists of `m1 * m4`, `m2 * m5`, and `m3 * m6`.


## Additively homomorphic encryption

Additively homomorphic encryption supports addition as a homomorphic operation.

```python
>>> c4 = add(params, c1, c2)
```

The decryption of `c4` is expected to be `m1 + m2`.
It can also compute the addition for 1D and 2D array-like ciphertexts.

```python
>>> c_v4 = add(params, c_v1, c_v2)
```

The decryption of `c_v4` is expected to be the array that consists of `m1 + m4`, `m2 + m5`, and `m3 + m6`.
Furthermore, additively homomorphic encryption supports integer multiplication, a product between a plaintext and a ciphertext.

```python
>>> c5 = int_mult(params, m1, c2)
>>> c_v5 = int_mult(params, [m1, m2, m3], c_v2)
```

The decryption of `c3` and `c_v3` are `m1 * m2` and the inner product of `[m1, m2, m3]` and `[m4, m5, m6]`, respectively.
Note that additively homomorphic encryption schemes in ECLib implement element-wise addition `elementwise_add()` and element-wise integer multiplication `elementwise_int_mult()` for convenience (see [API reference](../api_reference/api_reference.rst)).


## Fully homomorphic encryption

Fully homomorphic encryption supports both addition and multiplication as homomorphic operations.

```python
>>> c3 = mult(params, c1, c2)
>>> c4 = add(params, c1, c2)
```

Moreover, it supports integer multiplication.

```python
>>> c5 = int_mult(params, c1, c2)
```

The decryptions of `c3`, `c4`, and `c5` are expected to be coincide to ones in multiplicatively and additively homomorphic encryption.
The functions allows 1D and 2D array-like ciphertexts.

```python
>>> c_v3 = mult(params, c_v1, c_v2)
>>> c_v4 = add(params, c_v1, c_v2)
>>> c_v5 = int_mult(params, [m1, m2, m3], c_v2)
```

Similar to additively homomorphic encryption, fully homomorphic encryption schemes in ECLib implement element-wise addition `elementwise_add()`, element-wise integer multiplication `elementwise_int_mult()`, and element-wise multiplication `elementwise_mult()` for convenience (see [API reference](../api_reference/api_reference.rst)).