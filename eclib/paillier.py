#! /usr/bin/env python3

"""Paillier encryption scheme.

This module implements the Paillier encryption scheme, which is a public-key
cryptosystem based on the decisional composite residuosity assumption. It provides
functionalities for generating public and secret keys, encryption, decryption, and
homomorphic operations (addition and integer multiplication). It also includes
functions for encoding and decoding floating-point data into and from plaintexts.

Classes
-------
- PublicParameters
- SecretKey
- PublicKey

Functions
---------
- keygen
- encrypt
- decrypt
- add
- elementwise_add
- int_mult
- elementwise_int_mult
- encode
- decode
- enc
- dec
"""

from dataclasses import dataclass
from math import floor, gcd

import numpy as np
from numpy.typing import ArrayLike, NDArray

import eclib.primeutils as pu
import eclib.randutils as ru


@dataclass(slots=True)
class SecretKey:
    """
    Represents a secret key of the Paillier encryption scheme.

    Attributes
    ----------
    p : int
        The first prime factor.
    q : int
        The second prime factor.
    lmd : int
        Value of Euler's totient function of `n = p * q`.
    mu : int
        The multiplicative inverse of `lmd` modulo `n`.
    """

    p: int
    q: int
    lmd: int
    mu: int

    def __init__(self, bit_length: int):
        """
        Initializes a new SecretKey object.

        Parameters
        ----------
        bit_length : int
            Desired bit length of semiprime factors.
        """

        self.p, self.q = pu.get_semiprime_factors(bit_length)
        self.lmd = (self.p - 1) * (self.q - 1)
        self.mu = pow(self.lmd, -1, self.p * self.q)


@dataclass(slots=True)
class PublicParameters:
    """
    Represents public parameters of the Paillier encryption scheme.

    Attributes
    ----------
    n : int
        Product of two semiprime factors used as the modulus of plaintext space.
    n_square : int
        The square of `n` used as the modulus of ciphertext space.
    """

    n: int
    n_square: int

    def __init__(self, sk: SecretKey):
        """
        Initializes a new PublicParameters object.

        Parameters
        ----------
        sk : eclib.paillier.SecretKey
            Secret key used for computing the public parameters.

        See Also
        --------
        eclib.paillier.SecretKey
        """

        self.n = sk.p * sk.q
        self.n_square = self.n**2


@dataclass(slots=True)
class PublicKey:
    """
    Represents a public key of the Paillier encryption scheme.

    Attributes
    ----------
    g : int
        Public key value computed as `n + 1`, where `n` is the product of two semiprime
        factors used as the modulus of plaintext space.
    """

    g: int

    def __init__(self, params: PublicParameters):
        """
        Initializes a new PublicKey object.

        Parameters
        ----------
        params : eclib.paillier.PublicParameters
            Cryptosystem parameters used for computing the public key.

        See Also
        --------
        eclib.paillier.PublicParameters
        """

        self.g = params.n + 1


def keygen(bit_length: int) -> tuple[PublicParameters, PublicKey, SecretKey]:
    """
    Generates public parameters, a public key, and a secret key.

    Parameters
    ----------
    bit_length : int
        Desired bit length of semiprime factors whose product is used as the modulus of
        plaintext space, and the square of the product is used as the modulus of
        ciphertext space.

    Returns
    -------
    params : eclib.paillier.PublicParameters
        Cryptosystem parameters.
    pk : eclib.paillier.PublicKey
        Public key used for encryption.
    sk : eclib.paillier.SecretKey
        Secret key used for decryption.

    See Also
    --------
    eclib.paillier.PublicParameters
    eclib.paillier.PublicKey
    eclib.paillier.SecretKey
    """

    sk = SecretKey(bit_length)

    params = PublicParameters(sk)

    pk = PublicKey(params)

    return params, pk, sk


def encrypt(
    params: PublicParameters, pk: PublicKey, m: ArrayLike
) -> int | NDArray[np.object_]:
    """
    Encrypts a scalar, vector, or matrix plaintext `m` using a public key `pk`.

    Parameters
    ----------
    params : eclib.paillier.PublicParameters
        Cryptosystem parameters.
    pk : eclib.paillier.PublicKey
        Public key used for encryption.
    m : array_like
        Plaintext to be encrypted.

    Returns
    -------
    int or numpy.ndarray
        Ciphertext of the plaintext.

    Raises
    ------
    ValueError
        If the plaintext is not a scalar, vector, or matrix.

    See Also
    --------
    eclib.paillier.decrypt
    eclib.paillier.dec
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
    params: PublicParameters, sk: SecretKey, c: int | NDArray[np.object_]
) -> ArrayLike:
    """
    Decrypts a scalar, vector, or matrix ciphertext `c` using a secret key `sk`.

    Parameters
    ----------
    params : eclib.paillier.PublicParameters
        Cryptosystem parameters.
    sk : eclib.paillier.SecretKey
        Secret key used for decryption.
    c : int or numpy.ndarray
        Ciphertext to be decrypted.

    Returns
    -------
    array_like
        Decrypted plaintext.

    Raises
    ------
    ValueError
        If the ciphertext is not a scalar, vector, or matrix.

    See Also
    --------
    eclib.paillier.encrypt
    eclib.paillier.dec
    """

    c = np.asarray(c, dtype=object)

    match c.ndim:
        case 0:
            return _decrypt(params, sk, c.item())

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
    c1: int | NDArray[np.object_],
    c2: int | NDArray[np.object_],
) -> int | NDArray[np.object_]:
    """
    Computes a ciphertext of the addition of two scalar, vector, or matrix plaintexts
    corresponding to ciphertexts `c1` and `c2`.

    Parameters
    ----------
    params : eclib.paillier.PublicParameters
        Cryptosystem parameters.
    c1 : int or numpy.ndarray
        Ciphertext of the first plaintext.
    c2 : int or numpy.ndarray
        Ciphertext of the second plaintext.

    Returns
    -------
    int or numpy.ndarray
        Ciphertext of the addition of the plaintexts.

    Raises
    ------
    ValueError
        If the ciphertexts are not the following types of appropriate sizes:
        scalar-scalar, vector-vector, or matrix-matrix.

    See Also
    --------
    eclib.paillier.elementwise_add
    """

    c1 = np.asarray(c1, dtype=object)
    c2 = np.asarray(c2, dtype=object)

    if c1.shape == c2.shape:
        match c1.ndim:
            case 0:
                return _add(params, c1.item(), c2.item())

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
    c1: int | NDArray[np.object_],
    c2: int | NDArray[np.object_],
) -> int | NDArray[np.object_]:
    """
    Computes a ciphertext of the elementwise addition of two scalar, vector, or matrix
    plaintexts corresponding to ciphertexts `c1` and `c2`.

    Parameters
    ----------
    params : eclib.paillier.PublicParameters
        Cryptosystem parameters.
    c1 : int or numpy.ndarray
        Ciphertext of the first plaintext.
    c2 : int or numpy.ndarray
        Ciphertext of the second plaintext.

    Returns
    -------
    int or numpy.ndarray
        Ciphertext of the elementwise addition of the plaintexts.

    Raises
    ------
    ValueError
        If the ciphertexts are not the following types of appropriate sizes:
        scalar-scalar, vector-vector, matrix-vector, or matrix-matrix.

    See Also
    --------
    eclib.paillier.add
    """

    c1 = np.asarray(c1, dtype=object)
    c2 = np.asarray(c2, dtype=object)

    if c1.shape == c2.shape:
        return add(params, c1, c2)

    elif c1.ndim == 2 and c1.shape[1] == c2.shape[0]:
        return np.array(
            [
                [_add(params, c1[i][j], c2[j]) for j in range(c1.shape[1])]
                for i in range(c1.shape[0])
            ],
            dtype=object,
        )

    else:
        raise ValueError


def int_mult(
    params: PublicParameters, m: ArrayLike, c: int | NDArray[np.object_]
) -> int | NDArray[np.object_]:
    """
    Computes a ciphertext of the product of a scalar, vector, or matrix plaintext `m`
    and another scalar, vector, or matrix plaintext corresponding to a ciphertext `c`.

    Parameters
    ----------
    params : eclib.paillier.PublicParameters
        Cryptosystem parameters.
    m : array_like
        Plaintext to be multiplied.
    c : int or numpy.ndarray
        Ciphertext of a plaintext to be multiplied.

    Returns
    -------
    int or numpy.ndarray
        Ciphertext of the product of the plaintexts.

    Raises
    ------
    ValueError
        If the plaintext and ciphertext are not the following types of appropriate
        sizes: scalar-scalar, scalar-vector, scalar-matrix, vector-vector,
        matrix-vector, or matrix-matrix.

    See Also
    --------
    eclib.paillier.elementwise_int_mult
    """

    m = np.asarray(m, dtype=object)
    c = np.asarray(c, dtype=object)

    match m.ndim:
        case 0 if c.ndim == 0:
            return _int_mult(params, m.item(), c.item())

        case 0 if c.ndim == 1:
            return np.array(
                [_int_mult(params, m.item(), c[i]) for i in range(c.shape[0])],
                dtype=object,
            )

        case 0 if c.ndim == 2:
            return np.array(
                [
                    [_int_mult(params, m.item(), c[i][j]) for j in range(c.shape[1])]
                    for i in range(c.shape[0])
                ],
                dtype=object,
            )

        case 1 if c.ndim == 1 and m.shape[0] == c.shape[0]:
            c_s = 1

            for i in range(m.shape[0]):
                c_s = _add(params, c_s, _int_mult(params, m[i], c[i]))

            return c_s

        case 2 if c.ndim == 1 and m.shape[1] == c.shape[0]:
            c_v = np.ones(m.shape[0], dtype=object)

            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    c_v[i] = _add(params, c_v[i], _int_mult(params, m[i][j], c[j]))

            return c_v

        case 2 if c.ndim == 2 and m.shape[1] == c.shape[0]:
            c_m = np.ones([m.shape[0], c.shape[1]], dtype=object)

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
    params: PublicParameters, m: ArrayLike, c: int | NDArray[np.object_]
) -> int | NDArray[np.object_]:
    """
    Computes a ciphertext of the elementwise product of a scalar, vector, or matrix
    plaintext `m` and another scalar, vector, or matrix plaintext corresponding to a
    ciphertext `c`.

    Parameters
    ----------
    params : eclib.paillier.PublicParameters
        Cryptosystem parameters.
    m : array_like
        Plaintext to be multiplied.
    c : int or numpy.ndarray
        Ciphertext of a plaintext to be multiplied.

    Returns
    -------
    int or numpy.ndarray
        Ciphertext of the elementwise product of the plaintexts.

    Raises
    ------
    ValueError
        If the plaintext and ciphertext are not the following types of appropriate
        sizes: scalar-scalar, scalar-vector, scalar-matrix, vector-vector,
        matrix-vector, or matrix-matrix.

    See Also
    --------
    eclib.paillier.int_mult
    """

    m = np.asarray(m, dtype=object)
    c = np.asarray(c, dtype=object)

    match m.ndim:
        case 0:
            return int_mult(params, m.item(), c.item())

        case 1 if c.ndim == 1 and m.shape[0] == c.shape[0]:
            return np.array(
                [_int_mult(params, m[i], c[i]) for i in range(m.shape[0])],
                dtype=object,
            )

        case 2 if c.ndim == 1 and m.shape[1] == c.shape[0]:
            return np.array(
                [
                    [_int_mult(params, m[i][j], c[j]) for j in range(m.shape[1])]
                    for i in range(m.shape[0])
                ],
                dtype=object,
            )

        case 2 if c.ndim == 2 and m.shape[1] == c.shape[0]:
            return np.array(
                [
                    [_int_mult(params, m[i][j], c[i][j]) for j in range(m.shape[1])]
                    for i in range(m.shape[0])
                ],
                dtype=object,
            )

        case _:
            raise ValueError


def encode(params: PublicParameters, x: ArrayLike, scale: float) -> ArrayLike:
    """
    Encodes a scalar, vector, or matrix floating-point data `x` into a plaintext.

    Parameters
    ----------
    params : eclib.paillier.PublicParameters
        Cryptosystem parameters.
    x : array_like
        Floating-point data to be encoded.
    scale : float
        Scaling factor.

    Returns
    -------
    array_like
        Encoded plaintext.

    See Also
    --------
    eclib.paillier.decode
    eclib.paillier.enc
    """

    f = np.frompyfunc(_encode, 3, 1)
    return f(params, x, scale)


def decode(params: PublicParameters, m: ArrayLike, scale: float) -> ArrayLike:
    """
    Decodes a scalar, vector, or matrix plaintext `m` into a floating-point data.

    Parameters
    ----------
    params : eclib.paillier.PublicParameters
        Cryptosystem parameters.
    m : array_like
        Plaintext to be decoded.
    scale : float
        Scaling factor.

    Returns
    -------
    array_like
        Decoded floating-point data.

    See Also
    --------
    eclib.paillier.encode
    eclib.paillier.dec
    """

    f = np.frompyfunc(_decode, 3, 1)
    return f(params, m, scale)


def enc(
    params: PublicParameters, pk: PublicKey, x: ArrayLike, scale: float
) -> int | NDArray[np.object_]:
    """
    Encodes and encrypts a scalar, vector, or matrix floating-point data `x` using a
    public key `pk`.

    Parameters
    ----------
    params : eclib.paillier.PublicParameters
        Cryptosystem parameters.
    pk : eclib.paillier.PublicKey
        Public key used for encryption.
    x : array_like
        Floating-point data to be encoded and encrypted.
    scale : float
        Scaling factor.

    Returns
    -------
    int or numpy.ndarray
        Ciphertext of the encoded plaintext of the floating-point data.

    See Also
    --------
    eclib.paillier.encrypt
    eclib.paillier.encode
    """

    return encrypt(params, pk, encode(params, x, scale))


def dec(
    params: PublicParameters, sk: SecretKey, c: int | NDArray[np.object_], scale: float
) -> ArrayLike:
    """
    Decrypts and decodes a scalar, vector, or matrix ciphertext `c` using a secret key
    `sk`.

    Parameters
    ----------
    params : eclib.paillier.PublicParameters
        Cryptosystem parameters.
    sk : eclib.paillier.SecretKey
        Secret key used for decryption.
    c : int or numpy.ndarray
        Ciphertext to be decrypted and decoded.
    scale : float
        Scaling factor.

    Returns
    -------
    array_like
        Decoded floating-point data of the decrypted plaintext.

    See Also
    --------
    eclib.paillier.decrypt
    eclib.paillier.decode
    """

    return decode(params, decrypt(params, sk, c), scale)


def _encrypt(params: PublicParameters, pk: PublicKey, m: int) -> int:
    """
    Encrypts a message `m` using a public key `pk`.

    Parameters
    ----------
    params : eclib.paillier.PublicParameters
        Cryptosystem parameters.
    pk : eclib.paillier.PublicKey
        Public key used for encryption.
    m : int
        Plaintext to be encrypted.

    Returns
    -------
    int
        Ciphertext of the plaintext.
    """

    r = ru.get_rand(0, params.n)
    while gcd(r, params.n) != 1:
        r = ru.get_rand(0, params.n)

    return (
        pow(pk.g, m, params.n_square) * pow(r, params.n, params.n_square)
    ) % params.n_square


def _decrypt(params: PublicParameters, sk: SecretKey, c: int) -> int:
    """
    Decrypts a ciphertext `c` using a secret key `sk`.

    Parameters
    ----------
    params : eclib.paillier.PublicParameters
        Cryptosystem parameters.
    sk : eclib.paillier.SecretKey
        Secret key used for decryption.
    c : int
        Ciphertext to be decrypted.

    Returns
    -------
    int
        Decrypted plaintext.
    """

    return (((pow(c, sk.lmd, params.n_square) - 1) // params.n) * sk.mu) % params.n


def _add(params: PublicParameters, c1: int, c2: int) -> int:
    """
    Computes a ciphertext of the addition of two plaintexts corresponding to
    ciphertexts `c1` and `c2`.

    Parameters
    ----------
    params : eclib.paillier.PublicParameters
        Cryptosystem parameters.
    c1 : int
        Ciphertext of the first plaintext.
    c2 : int
        Ciphertext of the second plaintext.

    Returns
    -------
    int
        Ciphertext of the addition of the plaintexts.
    """

    return (c1 * c2) % params.n_square


def _int_mult(params: PublicParameters, m: int, c: int) -> int:
    """
    Computes a ciphertext of the product of a plaintext `m` and another plaintext
    corresponding to a ciphertext `c`.

    Parameters
    ----------
    params : eclib.paillier.PublicParameters
        Cryptosystem parameters.
    m : int
        Plaintext to be multiplied.
    c : int
        Ciphertext of a plaintext to be multiplied.

    Returns
    -------
    int
        Ciphertext of the product of the plaintexts.
    """

    return pow(c, m, params.n_square)


def _encode(params: PublicParameters, x: float, scale: float) -> int:
    """
    Encodes a floating-point number `x` into a plaintext.

    Parameters
    ----------
    params : eclib.paillier.PublicParameters
        Cryptosystem parameters.
    x : float
        Floating-point number to be encoded.
    scale : float
        Scaling factor.

    Returns
    -------
    int
        Encoded plaintext.

    Raises
    ------
    ValueError
        If the encoded value is out of range (underflow or overflow).
    """

    m = floor(x / scale + 0.5)

    if m < -((params.n - 1) // 2):
        raise ValueError("Underflow")

    elif m > (params.n // 2):
        raise ValueError("Overflow")

    else:
        return m % params.n


def _decode(params: PublicParameters, m: int, scale: float) -> float:
    """
    Decodes a plaintext `m` into a floating-point number.

    Parameters
    ----------
    params : eclib.paillier.PublicParameters
        Cryptosystem parameters.
    m : int
        Plaintext to be decoded.
    scale : float
        Scaling factor.

    Returns
    -------
    float
        Decoded floating-point number.
    """

    return (m - floor(m / params.n + 0.5) * params.n) * scale
