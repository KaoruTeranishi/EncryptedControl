#! /usr/bin/env python3

"""
gsw.py

This module implements the GSW encryption scheme, which is a fully homomorphic
encryption scheme based on the Learning with Errors (LWE) problem. It provides
functionalities for generating public parameters, public and secret keys, encryption,
decryption, and homomorphic operations (addition, multiplication, and integer
multiplication). It also includes functions for encoding and decoding floating-point
data into and from plaintexts.


Classes:
    PublicParameters: Represents public parameters of the GSW encryption scheme.

Functions:
    keygen: Generates public parameters, a public key, and a secret key.
    encrypt: Encrypts a scalar, vector, or matrix plaintext.
    decrypt: Decrypts a scalar, vector, or matrix ciphertext.
    add: Computes a ciphertext of the addition of two scalar, vector, or matrix
        plaintexts.
    elementwise_add: Computes a ciphertext of the elementwise addition of two scalar,
        vector, or matrix plaintexts.
    mult: Computes a ciphertext of the product of two scalar, vector, or matrix
        plaintexts.
    elementwise_mult: Computes a ciphertext of the elementwise product of two scalar,
        vector, or matrix plaintexts.
    int_mult: Computes a ciphertext of the product of a scalar, vector, or matrix
        plaintext and another scalar, vector, or matrix plaintext.
    elementwise_int_mult: Computes a ciphertext of the elementwise product of a scalar,
        vector, or matrix plaintext and another scalar, vector, or matrix plaintext.
    encode: Encodes a scalar, vector, or matrix floating-point data into a plaintext.
    decode: Decodes a scalar, vector, or matrix plaintext into floating-point data.
    enc: Encodes and encrypts a scalar, vector, or matrix floating-point data.
    dec: Decrypts and decodes a scalar, vector, or matrix ciphertext.

Dependencies:
    numpy: Fundamental package for scientific computing with Python.
    numpy.typing: Type hints for NumPy.
    eclib.randutils: Utility functions for generating random numbers.
    eclib.regev: Regev (LWE) encryption scheme.
"""

from dataclasses import dataclass
from math import ceil, floor, log2
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

import eclib.randutils as ru
from eclib import regev
from eclib.regev import PublicKey, SecretKey


@dataclass(slots=True)
class PublicParameters:
    """
    Represents public parameters of the GSW encryption scheme.

    Attributes:
        n (int): Dimension of a the lattice, which is equal to the dimension of secret
            key.
        q (int): Modulus of plaintext and ciphertext spaces.
        sigma (float): Standard deviation of the discrete Gaussian distribution with
            mean zero used as an error distribution.
        m (int): Subdimension of lattice.
        l (int): Bit length of the modulus.
        N (int): N = (n + 1) * l.
    """

    n: int
    q: int
    sigma: float
    m: int
    l: int
    N: int

    def __init__(self, n: int, q: int, sigma: float, m: Optional[int] = None):
        """
        Initializes a new PublicParameters object.

        Args:
            n (int): Dimension of a lattice, which is equal to the dimension of secret
                key.
            q (int): Modulus of plaintext and ciphertext spaces.
            sigma (float): Standard deviation of the discrete Gaussian distribution
                with mean zero used as an error distribution.
            m (int, optional, default = None): Subdimension of the lattice.

        Note:
            If `m` is not provided, it is set to `2 * n * ceil(log2(q))`.
        """

        self.n = n
        self.q = q
        self.sigma = sigma
        self.l = ceil(log2(q))
        self.N = (n + 1) * self.l

        if m is None:
            self.m = 2 * n * ceil(log2(q))

        else:
            self.m = m


def keygen(n: int, q: int, sigma: float, m: Optional[int] = None):
    """
    Generates public parameters, a public key, and a secret key.

    Args:
        n (int): Dimension of a lattice, which is equal to the dimension of secret key.
        q (int): Modulus of plaintext and ciphertext spaces.
        sigma (float): Standard deviation of the discrete Gaussian distribution with
            mean zero used as an error distribution.
        m (int, optional, default = None): Subdimension of the lattice.

    Returns:
        tuple[PublicParameters, PublicKey, SecretKey]: Tuple containing the public
            parameters and Regev (LWE) public and secret keys.

    Note:
        If `m` is not provided, it is set to `2 * n * ceil(log2(q))`.

    See Also:
        PublicParameters
        :class:`~regev.PublicKey`
        :class:`~regev.SecretKey`
    """

    params = PublicParameters(n, q, sigma, m)
    lwe_params = regev.PublicParameters(n, 0, q, sigma, m)

    sk = SecretKey(lwe_params)

    pk = PublicKey(lwe_params, sk)

    return params, pk, sk


def encrypt(
    params: PublicParameters, pk: PublicKey, m: ArrayLike
) -> NDArray[np.object_]:
    """
    Encrypts a scalar, vector, or matrix plaintext `m` using a public key `pk`.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        pk (PublicKey): Public key used for encryption.
        m (ArrayLike): Plaintext to be encrypted.

    Returns:
        NDArray[np.object_]: Ciphertext of the plaintext.

    Raises:
        ValueError: If the plaintext is not a scalar, vector, or matrix.

    See Also:
        decrypt
        enc
    """

    m = np.asarray(m, dtype=object)

    match m.ndim:
        case 0:
            return _encrypt(params, pk, m.item())

        case 1:
            return np.array(
                [_encrypt(params, pk, m[i]) for i in range(m.shape[0])],
                dtype=object,
            )

        case 2:
            return np.array(
                [
                    [_encrypt(params, pk, m[i][j]) for j in range(m.shape[1])]
                    for i in range(m.shape[0])
                ],
                dtype=object,
            )

        case _:
            raise ValueError


def decrypt(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_]
) -> ArrayLike:
    """
    Decrypts a scalar, vector, or matrix ciphertext `c` using a secret key `sk`.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        sk (SecretKey): Secret key used for decryption.
        c (NDArray[np.object_]): Ciphertext to be decrypted.

    Returns:
        ArrayLike: Decrypted plaintext.

    Raises:
        ValueError: If the ciphertext is not a scalar, vector, or matrix.

    See Also:
        encrypt
        dec
    """

    c = np.asarray(c, dtype=object)

    match c.ndim - 2:
        case 0:
            return _decrypt(params, sk, c)

        case 1:
            return np.array(
                [_decrypt(params, sk, c[i]) for i in range(c.shape[0])],
                dtype=object,
            )

        case 2:
            return np.array(
                [
                    [_decrypt(params, sk, c[i][j]) for j in range(c.shape[1])]
                    for i in range(c.shape[0])
                ],
                dtype=object,
            )

        case _:
            raise ValueError


def add(
    params: PublicParameters,
    c1: NDArray[np.object_],
    c2: NDArray[np.object_],
) -> NDArray[np.object_]:
    """
    Computes a ciphertext of the addition of two scalar, vector, or matrix plaintexts
    corresponding to ciphertexts `c1` and `c2`.

    Args:
        params (PublicParameters): The public parameters of the Paillier cryptosystem.
        c1 (NDArray[np.object_]): Ciphertext of the first plaintext.
        c2 (NDArray[np.object_]): Ciphertext of the second plaintext.

    Returns:
        NDArray[np.object_]: Ciphertext of the addition of the plaintexts.

    Raises:
        ValueError: If the ciphertexts are not the following types of appropriate
            sizes: scalar-scalar, vector-vector, or matrix-matrix.

    See Also:
        elementwise_add
    """

    c1 = np.asarray(c1, dtype=object)
    c2 = np.asarray(c2, dtype=object)

    if c1.shape == c2.shape:
        match c1.ndim - 2:
            case 0:
                return _add(params, c1, c2)

            case 1:
                return np.array(
                    [_add(params, c1[i], c2[i]) for i in range(c1.shape[0])],
                    dtype=object,
                )

            case 2:
                return np.array(
                    [
                        [_add(params, c1[i][j], c2[i][j]) for j in range(c1.shape[1])]
                        for i in range(c1.shape[0])
                    ],
                    dtype=object,
                )

            case _:
                raise ValueError

    else:
        raise ValueError


def elementwise_add(
    params: PublicParameters,
    c1: NDArray[np.object_],
    c2: NDArray[np.object_],
) -> NDArray[np.object_]:
    """
    Computes a ciphertext of the elementwise addition of two scalar, vector, or matrix
    plaintexts corresponding to ciphertexts `c1` and `c2`.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        c1 (NDArray[np.object_]): Ciphertext of the first plaintext.
        c2 (NDArray[np.object_]): Ciphertext of the second plaintext.

    Returns:
        NDArray[np.object_]: Ciphertext of the elementwise addition of the plaintexts.

    Raises:
        ValueError: If the ciphertexts are not the following types of appropriate
            sizes: scalar-scalar, vector-vector, matrix-vector, or matrix-matrix.

    See Also:
        add
    """

    c1 = np.asarray(c1, dtype=object)
    c2 = np.asarray(c2, dtype=object)

    if c1.shape == c2.shape:
        return add(params, c1, c2)

    elif c1.ndim - 2 == 2 and c1.shape[1] == c2.shape[0]:
        return np.array(
            [
                [_add(params, c1[i][j], c2[j]) for j in range(c1.shape[1])]
                for i in range(c1.shape[0])
            ],
            dtype=object,
        )

    else:
        raise ValueError


def mult(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    Computes a ciphertext of the product of two scalar, vector, or matrix plaintexts
    corresponding to ciphertexts `c1` and `c2`.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        c1 (NDArray[np.object_]): Ciphertext of the first plaintext.
        c2 (NDArray[np.object_]): Ciphertext of the second plaintext.

    Returns:
        NDArray[np.object_]: Ciphertext of the product of the plaintexts.

    Raises:
        ValueError: If the ciphertexts are not the following types of appropriate
        sizes: scalar-scalar, scalar-vector, scalar-matrix, vector-vector,
        matrix-vector, or matrix-matrix.

    See Also:
        elementwise_mult
    """

    c1 = np.asarray(c1, dtype=object)
    c2 = np.asarray(c2, dtype=object)

    match c1.ndim - 2:
        case 0 if c2.ndim - 2 == 0:
            return _mult(params, c1, c2)

        case 0 if c2.ndim - 2 == 1:
            return np.array(
                [_mult(params, c1, c2[i]) for i in range(c2.shape[0])],
                dtype=object,
            )

        case 0 if c2.ndim - 2 == 2:
            return np.array(
                [
                    [_mult(params, c1, c2[i][j]) for j in range(c2.shape[1])]
                    for i in range(c2.shape[0])
                ],
                dtype=object,
            )

        case 1 if c2.ndim - 2 == 1 and c1.shape[0] == c2.shape[0]:
            c_s = np.zeros([params.n + 1, params.N], dtype=object)

            for i in range(c1.shape[0]):
                c_s = _add(params, c_s, _mult(params, c1[i], c2[i]))

            return c_s

        case 2 if c2.ndim - 2 == 1 and c1.shape[1] == c2.shape[0]:
            c_v = np.zeros([c1.shape[0], params.n + 1, params.N], dtype=object)

            for i in range(c1.shape[0]):
                for j in range(c1.shape[1]):
                    c_v[i] = _add(params, c_v[i], _mult(params, c1[i][j], c2[j]))

            return c_v

        case 2 if c2.ndim - 2 == 2 and c1.shape[1] == c2.shape[0]:
            c_m = np.zeros(
                [c1.shape[0], c2.shape[1], params.n + 1, params.N], dtype=object
            )

            for i in range(c1.shape[0]):
                for j in range(c2.shape[1]):
                    for k in range(c1.shape[1]):
                        c_m[i][j] = _add(
                            params, c_m[i][j], _mult(params, c1[i][k], c2[k][j])
                        )

            return c_m

        case _:
            raise ValueError


def elementwise_mult(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    Computes a ciphertext of the elementwise product of two scalar, vector, or matrix
    plaintexts corresponding to ciphertexts `c1` and `c2`.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        c1 (NDArray[np.object_]): Ciphertext of the first plaintext.
        c2 (NDArray[np.object_]): Ciphertext of the second plaintext.

    Returns:
        NDArray[np.object_]: Ciphertext of the elementwise product of the plaintexts.

    Raises:
        ValueError: If the ciphertexts are not the following types of appropriate
        sizes: scalar-scalar, scalar-vector, scalar-matrix, vector-vector,
        matrix-vector, or matrix-matrix.

    See Also:
        mult
    """

    c1 = np.asarray(c1, dtype=object)
    c2 = np.asarray(c2, dtype=object)

    match c1.ndim - 2:
        case 0:
            return mult(params, c1, c2)

        case 1 if c2.ndim - 2 == 1 and c1.shape[0] == c2.shape[0]:
            return np.array(
                [_mult(params, c1[i], c2[i]) for i in range(c1.shape[0])],
                dtype=object,
            )

        case 2 if c2.ndim - 2 == 1 and c1.shape[1] == c2.shape[0]:
            return np.array(
                [
                    [_mult(params, c1[i][j], c2[j]) for j in range(c1.shape[1])]
                    for i in range(c1.shape[0])
                ],
                dtype=object,
            )

        case 2 if c2.ndim - 2 == 2 and c1.shape[1] == c2.shape[0]:
            return np.array(
                [
                    [_mult(params, c1[i][j], c2[i][j]) for j in range(c1.shape[1])]
                    for i in range(c1.shape[0])
                ],
                dtype=object,
            )

        case _:
            raise ValueError


def int_mult(
    params: PublicParameters, m: ArrayLike, c: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    Computes a ciphertext of the product of a scalar, vector, or matrix plaintext `m`
    and another scalar, vector, or matrix plaintext corresponding to a ciphertext `c`.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        m (ArrayLike): Plaintext to be multiplied.
        c (NDArray[np.object_]): Ciphertext of a plaintext to be multiplied.

    Returns:
        NDArray[np.object_]: Ciphertext of the product of the plaintexts.

    Raises:
        ValueError: If the plaintext and ciphertext are not the following types of
            appropriate sizes: scalar-scalar, scalar-vector, scalar-matrix,
            vector-vector, matrix-vector, or matrix-matrix.

    See Also:
        elementwise_int_mult
    """

    m = np.asarray(m, dtype=object)
    c = np.asarray(c, dtype=object)

    match m.ndim:
        case 0 if c.ndim - 2 == 0:
            return _int_mult(params, m.item(), c)

        case 0 if c.ndim - 2 == 1:
            return np.array(
                [_int_mult(params, m.item(), c[i]) for i in range(c.shape[0])],
                dtype=object,
            )

        case 0 if c.ndim - 2 == 2:
            return np.array(
                [
                    [_int_mult(params, m.item(), c[i][j]) for j in range(c.shape[1])]
                    for i in range(c.shape[0])
                ],
                dtype=object,
            )

        case 1 if c.ndim - 2 == 1 and m.shape[0] == c.shape[0]:
            c_s = np.zeros([params.n + 1, params.N], dtype=object)

            for i in range(m.shape[0]):
                c_s = _add(params, c_s, _int_mult(params, m[i], c[i]))

            return c_s

        case 2 if c.ndim - 2 == 1 and m.shape[1] == c.shape[0]:
            c_v = np.zeros([m.shape[0], params.n + 1, params.N], dtype=object)

            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    c_v[i] = _add(params, c_v[i], _int_mult(params, m[i][j], c[j]))

            return c_v

        case 2 if c.ndim - 2 == 2 and m.shape[1] == c.shape[0]:
            c_m = np.zeros(
                [m.shape[0], c.shape[1], params.n + 1, params.N], dtype=object
            )

            for i in range(m.shape[0]):
                for j in range(c.shape[1]):
                    for k in range(m.shape[1]):
                        c_m[i][j] = _add(
                            params, c_m[i][j], _int_mult(params, m[i][k], c[k][j])
                        )

            return c_m

        case _:
            raise ValueError


def elementwise_int_mult(
    params: PublicParameters, m: ArrayLike, c: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    Computes a ciphertext of the elementwise product of a scalar, vector, or matrix
    plaintext `m` and another scalar, vector, or matrix plaintext corresponding to a
    ciphertext `c`.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        m (ArrayLike): Plaintext to be multiplied.
        c (NDArray[np.object_]): Ciphertext of a plaintext to be multiplied.

    Returns:
        NDArray[np.object_]: Ciphertext of the elementwise product of the plaintexts.

    Raises:
        ValueError: If the plaintext and ciphertext are not the following types of
            appropriate sizes: scalar-scalar, scalar-vector, scalar-matrix,
            vector-vector, matrix-vector, or matrix-matrix.

    See Also:
        int_mult
    """

    m = np.asarray(m, dtype=object)
    c = np.asarray(c, dtype=object)

    match m.ndim:
        case 0:
            return int_mult(params, m.item(), c)

        case 1 if c.ndim - 2 == 1 and m.shape[0] == c.shape[0]:
            return np.array(
                [_int_mult(params, m[i], c[i]) for i in range(m.shape[0])],
                dtype=object,
            )

        case 2 if c.ndim - 2 == 1 and m.shape[1] == c.shape[0]:
            return np.array(
                [
                    [_int_mult(params, m[i][j], c[j]) for j in range(m.shape[1])]
                    for i in range(c.shape[0])
                ],
                dtype=object,
            )

        case 2 if c.ndim - 2 == 2 and m.shape[1] == c.shape[0]:
            return np.array(
                [
                    [_int_mult(params, m[i][j], c[i][j]) for j in range(m.shape[1])]
                    for i in range(m.shape[0])
                ],
                dtype=object,
            )

        case _:
            raise ValueError


def encode(params: PublicParameters, x: ArrayLike, delta: float) -> ArrayLike:
    """
    Encodes a scalar, vector, or matrix floating-point data `x` into a plaintext.

    Parameters:
        params (PublicParameters): Cryptosystem parameters.
        x (ArrayLike): Floating-point data to be encoded.
        delta (float): Scaling factor.

    Returns:
        ArrayLike: Encoded plaintext.

    See Also:
        decode
        enc
    """

    f = np.frompyfunc(_encode, 3, 1)
    return f(params, x, delta)


def decode(params: PublicParameters, m: ArrayLike, delta: float) -> ArrayLike:
    """
    Decodes a scalar, vector, or matrix plaintext `m` into a floating-point data.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        m (ArrayLike): Plaintext to be decoded.
        delta (float): Scaling factor.

    Returns:
        ArrayLike: Decoded floating-point data.

    See Also:
        encode
        dec
    """

    f = np.frompyfunc(_decode, 3, 1)
    return f(params, m, delta)


def enc(
    params: PublicParameters, pk: PublicKey, x: ArrayLike, delta: float
) -> NDArray[np.object_]:
    """
    Encodes and encrypts a scalar, vector, or matrix floating-point data `x` using a
    public key `pk`.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        pk (PublicKey): Public key used for encryption.
        x (ArrayLike): Floating-point data to be encoded and encrypted.
        delta (float): Scaling factor.

    Returns:
        NDArray[np.object_]: Ciphertext of the encoded plaintext of the floating-point
            data.

    See Also:
        encrypt
        encode
    """

    return encrypt(params, pk, encode(params, x, delta))


def dec(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_], delta: float
) -> ArrayLike:
    """
    Decrypts and decodes a scalar, vector, or matrix ciphertext `c` using a secret key
    `sk`.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        sk (SecretKey): Secret key used for decryption.
        c (NDArray[np.object_]): Ciphertext to be decrypted and decoded.
        delta (float): Scaling factor.

    Returns:
        ArrayLike: Decoded floating-point data of the decrypted plaintext.

    See Also:
        decrypt
        decode
    """

    return decode(params, decrypt(params, sk, c), delta)


def _encrypt(params: PublicParameters, pk: PublicKey, m: int) -> NDArray[np.object_]:
    """
    Encrypts a message `m` using a public key `pk`.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        pk (PublicKey): Public key used for encryption.
        m (int): Plaintext to be encrypted.

    Returns:
        NDArray[np.object_]: Ciphertext of the plaintext.
    """

    R = np.array(
        [[ru.get_rand(0, 2) for _ in range(params.N)] for _ in range(params.m)],
        dtype=object,
    )
    G = _gadget(params)

    return (pk.B @ R + m * G) % params.q


def _decrypt(params: PublicParameters, sk: SecretKey, c: NDArray[np.object_]) -> int:
    """
    Decrypts a ciphertext `c` using a secret key `sk`.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        sk (SecretKey): Secret key used for decryption.
        c (NDArray[np.object_]): Ciphertext to be decrypted.

    Returns:
        int: Decrypted plaintext.
    """

    tmp = (np.block([1, -sk.s.T]) @ c[:, 0 : params.l]).reshape(-1) % params.q

    bits = np.zeros(params.l, dtype=object)
    lsb_sum = 0

    bits[0] = floor(tmp[params.l - 1] / pow(2, params.l - 1) + 0.5) % 2
    m = bits[0]

    for i in range(1, params.l):
        lsb_sum = lsb_sum / 2 + pow(2, params.l - 2) * bits[i - 1]
        bits[i] = (
            floor((tmp[params.l - 1 - i] - lsb_sum) / pow(2, params.l - 1) + 0.5) % 2
        )
        m += (pow(2, i) * bits[i]) % params.q

    return m


def _add(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    Computes a ciphertext of the addition of two plaintexts corresponding to
    ciphertexts `c1` and `c2`.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        c1 (NDArray[np.object_]): Ciphertext of the first plaintext.
        c2 (NDArray[np.object_]): Ciphertext of the second plaintext.

    Returns:
        NDArray[np.object_]: Ciphertext of the addition of the plaintexts.
    """

    return (c1 + c2) % params.q


def _mult(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    Computes a ciphertext of the product of two plaintexts corresponding to ciphertexts
    `c1` and `c2`.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        c1 (NDArray[np.object_]): Ciphertext of the first plaintext.
        c2 (NDArray[np.object_]): Ciphertext of the second plaintext.

    Returns:
        NDArray[np.object_]: Ciphertext of the product of the plaintexts.
    """

    bit_decomposed = np.block(
        [_bitdecomp(params, c2[:, i]) for i in range(c2.shape[1])]
    )
    return (c1 @ bit_decomposed) % params.q


def _int_mult(
    params: PublicParameters, m: int, c: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    Computes a ciphertext of the product of a plaintext `m` and another plaintext
    corresponding to a ciphertext `c`.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        m (int): Plaintext to be multiplied.
        c (NDArray[np.object_]): Ciphertext of a plaintext to be multiplied.

    Returns:
        NDArray[np.object_]: Ciphertext of the product of the plaintexts.
    """

    return (m * c) % params.q


def _encode(params: PublicParameters, x: float, delta: float) -> int:
    """
    Encodes a floating-point number `x` into a plaintext.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        x (float): Floating-point number to be encoded.
        delta (float): Scaling factor.

    Returns:
        int: Encoded plaintext.

    Raises:
        ValueError: If the encoded value is out of range (underflow or overflow).
    """

    m = floor(x / delta + 0.5)

    if m < -((params.q - 1) // 2):
        raise ValueError("Underflow")

    elif m > (params.q // 2):
        raise ValueError("Overflow")

    else:
        return m % params.q


def _decode(params: PublicParameters, m: int, delta: float) -> float:
    """
    Decodes a plaintext `m` into a floating-point number.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        m (int): Plaintext to be decoded.
        delta (float): Scaling factor.

    Returns:
        float: Decoded floating-point number.
    """

    return (m - floor(m / params.q + 0.5) * params.q) * delta


def _gadget(params: PublicParameters) -> NDArray[np.object_]:
    """
    Constructs the gadget matrix for the GSW encryption scheme.

    Parameters:
        params (PublicParameters): Cryptosystem parameters.

    Returns:
        NDArray[np.object_]: The gadget matrix.

    """

    g = 2 ** np.arange(params.l, dtype=object)
    return np.kron(np.identity(params.n + 1, dtype=object), g)


def _bitdecomp(params: PublicParameters, v: ArrayLike) -> NDArray[np.object_]:
    """
    Decomposes the input array `v` into its binary representation.

    Args:
        params (PublicParameters): Cryptosystem parameters.
        v (ArrayLike): Input array to be decomposed.

    Returns:
        NDArray[np.object_]: Binary decomposition of the input array.

    Raises:
        ValueError: If the input array is not one-dimensional.
    """

    v = np.asarray(v, dtype=object)

    if v.ndim != 1:
        raise ValueError("Input array must be one-dimensional.")

    else:
        le_bin_repr = [np.binary_repr(x, params.l)[::-1] for x in v]

        return np.array(
            [
                [int(le_bin_repr[i][j])]
                for i in range(len(le_bin_repr))
                for j in range(params.l)
            ],
            dtype=object,
        )
