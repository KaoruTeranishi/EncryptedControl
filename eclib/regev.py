#! /usr/bin/env python3

"""Regev (LWE) encryption scheme.

This module implements the Regev (LWE) encryption scheme, which is a public-key
cryptosystem based on the Learning With Errors (LWE) problem. It provides
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
from math import ceil, floor, log2
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

import eclib.randutils as ru


@dataclass(slots=True)
class PublicParameters:
    """
    Represents public parameters of the Regev (LWE) encryption scheme.

    Attributes
    ----------
    n : int
        Dimension of a lattice, which is equal to the dimension of secret key.
    t : int
        Modulus of a plaintext space.
    q : int
        Modulus of a ciphertext space.
    sigma : float
        Standard deviation of the discrete Gaussian distribution with mean zero used as
        an error distribution.
    m : int
        Subdimension of the lattice.
    """

    n: int
    t: int
    q: int
    sigma: float
    m: int

    def __init__(self, n: int, t: int, q: int, sigma: float, m: Optional[int] = None):
        """
        Initializes a new PublicParameters object.

        Parameters
        ----------
        n : int
            Dimension of a lattice, which is equal to the dimension of secret key.
        t : int
            Modulus of a plaintext space.
        q : int
            Modulus of a ciphertext space.
        sigma : float
            Standard deviation of the discrete Gaussian distribution with mean zero used
            as an error distribution.
        m : int, optional
            Subdimension of the lattice.

        Note
        ----
        If `m` is not provided, it is set to `2 * n * ceil(log2(q))`.
        """

        self.n = n
        self.t = t
        self.q = q
        self.sigma = sigma

        if m is None:
            self.m = 2 * n * ceil(log2(q))
        else:
            self.m = m


@dataclass(slots=True)
class SecretKey:
    """
    Represents a secret key of the Regev (LWE) encryption scheme.

    Attributes
    ----------
    s : numpy.ndarray
        Secret key value.
    """

    s: NDArray[np.object_]

    def __init__(self, params: PublicParameters):
        """
        Initializes a new SecretKey object.

        Parameters
        ----------
        params : eclib.regev.PublicParameters
            Cryptosystem parameters.

        Note
        ----
        A secret key is a `n`-dimensional random vector of integers modulo `q`, which
        is the modulus of a ciphertext space.

        See Also
        --------
        eclib.regev.PublicParameters
        """

        self.s = np.array(
            [[ru.get_rand(0, params.q)] for _ in range(params.n)],
            dtype=object,
        )


@dataclass(slots=True)
class PublicKey:
    """
    Represents a public key of the Regev (LWE) encryption scheme.

    Attributes
    ----------
    A : numpy.ndarray
        Public key matrix.
    b : numpy.ndarray
        Public key vector.
    B : numpy.ndarray
        Concatenation of the vector `b` and the matrix `A`.
    """

    A: NDArray[np.object_]
    b: NDArray[np.object_]
    B: NDArray[np.object_]

    def __init__(self, params: PublicParameters, sk: SecretKey):
        """
        Initializes a new PublicKey object.

        Parameters
        ----------
        params : eclib.regev.PublicParameters
            Cryptosystem parameters.
        sk : eclib.regev.SecretKey
            Secret key used for computing the public key.

        Note
        ----
        The public key is a matrix `B`, which is a concatenation of a `m`-dimensional
        row vector `b` and a `n`-by-`m` matrix `A`. The matrix `A` is a random matrix of
        integers modulo `q`, and the vector `b` is given by `b = s^T A + e^T mod q`,
        where `q` is the modulus of a ciphertext space, `s` is the secret key, and `e`
        is a `m`-dimensional error vector sampled from the discrete Gaussian
        distribution with mean zero and standard deviation `sigma`.

        See Also
        --------
        eclib.regev.PublicParameters
        eclib.regev.SecretKey
        """

        self.A = np.array(
            [
                [ru.get_rand(0, params.q) for _ in range(params.m)]
                for _ in range(params.n)
            ],
            dtype=object,
        )

        e = np.array(
            ru.get_int_gaussian(0, params.sigma, params.m),
            dtype=object,
        ).reshape(-1, 1)

        self.b = (sk.s.T @ self.A + e.T) % params.q

        self.B = np.block([[self.b], [self.A]])


def keygen(n: int, t: int, q: int, sigma: float, m: Optional[int] = None):
    """
    Generates public parameters, a public key, and a secret key.

    Parameters
    ----------
    n : int
        Dimension of a lattice, which is equal to the dimension of secret key.
    t : int
        Modulus of a plaintext space.
    q : int
        Modulus of a ciphertext space.
    sigma : float
        Standard deviation of the discrete Gaussian distribution with mean zero used as
        an error distribution.
    m : int, optional
        Subdimension of the lattice.

    Returns
    -------
    params : eclib.regev.PublicParameters
        Cryptosystem parameters.
    pk : eclib.regev.PublicKey
        Public key used for encryption.
    sk : eclib.regev.SecretKey
        Secret key used for decryption.

    Note
    ----
    If `m` is not provided, it is set to `2 * n * ceil(log2(q))`.

    See Also
    --------
    eclib.regev.PublicParameters
    eclib.regev.PublicKey
    eclib.regev.SecretKey
    """

    params = PublicParameters(n, t, q, sigma, m)

    sk = SecretKey(params)

    pk = PublicKey(params, sk)

    return params, pk, sk


def encrypt(
    params: PublicParameters, pk: PublicKey, m: ArrayLike
) -> NDArray[np.object_]:
    """
    Encrypts a scalar, vector, or matrix plaintext `m` using a public key `pk`.

    Parameters
    ----------
    params : eclib.regev.PublicParameters
        Cryptosystem parameters.
    pk : eclib.regev.PublicKey
        Public key used for encryption.
    m : array_like
        Plaintext to be encrypted.

    Returns
    -------
    numpy.ndarray
        Ciphertext of the plaintext.

    Raises
    ------
    ValueError
        If the plaintext is not a scalar, vector, or matrix.

    See Also
    --------
    eclib.regev.decrypt
    eclib.regev.enc
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

    Parameters
    ----------
    params : eclib.regev.PublicParameters
        Cryptosystem parameters.
    sk : eclib.regev.SecretKey
        Secret key used for decryption.
    c : numpy.ndarray
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
    eclib.regev.encrypt
    eclib.regev.dec
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

    Parameters
    ----------
    params : eclib.regev.PublicParameters
        Cryptosystem parameters.
    c1 : numpy.ndarray
        Ciphertext of the first plaintext.
    c2 : numpy.ndarray
        Ciphertext of the second plaintext.

    Returns
    -------
    numpy.ndarray
        Ciphertext of the addition of the plaintexts.

    Raises
    ------
    ValueError
        If the ciphertexts are not the following types of appropriate sizes:
        scalar-scalar, vector-vector, or matrix-matrix.

    See Also
    --------
    eclib.regev.elementwise_add
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

    Parameters
    ----------
    params : eclib.regev.PublicParameters
        Cryptosystem parameters.
    c1 : numpy.ndarray
        Ciphertext of the first plaintext.
    c2 : numpy.ndarray
        Ciphertext of the second plaintext.

    Returns
    -------
    numpy.ndarray
        Ciphertext of the elementwise addition of the plaintexts.

    Raises
    ------
    ValueError
        If the ciphertexts are not the following types of appropriate sizes:
        scalar-scalar, vector-vector, matrix-vector, or matrix-matrix.

    See Also
    --------
    eclib.regev.add
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


def int_mult(
    params: PublicParameters, m: ArrayLike, c: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    Computes a ciphertext of the product of a scalar, vector, or matrix plaintext `m`
    and another scalar, vector, or matrix plaintext corresponding to a ciphertext `c`.

    Parameters
    ----------
    params : eclib.regev.PublicParameters
        Cryptosystem parameters.
    m : array_like
        Plaintext to be multiplied.
    c : numpy.ndarray
        Ciphertext of a plaintext to be multiplied.

    Returns
    -------
    numpy.ndarray
        Ciphertext of the product of the plaintexts.

    Raises
    ------
    ValueError
        If the plaintext and ciphertext are not the following types of appropriate
        sizes: scalar-scalar, scalar-vector, scalar-matrix, vector-vector,
        matrix-vector, or matrix-matrix.

    See Also
    --------
    eclib.regev.elementwise_int_mult
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
            c_s = np.zeros([params.n + 1, 1], dtype=object)

            for i in range(m.shape[0]):
                c_s = _add(params, c_s, _int_mult(params, m[i], c[i]))

            return c_s

        case 2 if c.ndim - 2 == 1 and m.shape[1] == c.shape[0]:
            c_v = np.zeros([m.shape[0], params.n + 1, 1], dtype=object)

            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    c_v[i] = _add(params, c_v[i], _int_mult(params, m[i][j], c[j]))

            return c_v

        case 2 if c.ndim - 2 == 2 and m.shape[1] == c.shape[0]:
            c_m = np.zeros([m.shape[0], c.shape[1], params.n + 1, 1], dtype=object)

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

    Parameters
    ----------
    params : eclib.regev.PublicParameters
        Cryptosystem parameters.
    m : array_like
        Plaintext to be multiplied.
    c : numpy.ndarray
        Ciphertext of a plaintext to be multiplied.

    Returns
    -------
    numpy.ndarray
        Ciphertext of the elementwise product of the plaintexts.

    Raises
    ------
    ValueError
        If the plaintext and ciphertext are not the following types of appropriate
        sizes: scalar-scalar, scalar-vector, scalar-matrix, vector-vector,
        matrix-vector, or matrix-matrix.

    See Also
    --------
    eclib.regev.int_mult
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
                    for i in range(m.shape[0])
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


def encode(params: PublicParameters, x: ArrayLike, scale: float) -> ArrayLike:
    """
    Encodes a scalar, vector, or matrix floating-point data `x` into a plaintext.

    Parameters
    ----------
    params : eclib.regev.PublicParameters
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
    eclib.regev.decode
    eclib.regev.enc
    """

    f = np.frompyfunc(_encode, 3, 1)
    return f(params, x, scale)


def decode(params: PublicParameters, m: ArrayLike, scale: float) -> ArrayLike:
    """
    Decodes a scalar, vector, or matrix plaintext `m` into a floating-point data.

    Parameters
    ----------
    params : eclib.regev.PublicParameters
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
    eclib.regev.encode
    eclib.regev.dec
    """

    f = np.frompyfunc(_decode, 3, 1)
    return f(params, m, scale)


def enc(
    params: PublicParameters, pk: PublicKey, x: ArrayLike, scale: float
) -> NDArray[np.object_]:
    """
    Encodes and encrypts a scalar, vector, or matrix floating-point data `x` using a
    public key `pk`.

    Parameters
    ----------
    params : eclib.regev.PublicParameters
        Cryptosystem parameters.
    pk : eclib.regev.PublicKey
        Public key used for encryption.
    x : array_like
        Floating-point data to be encoded and encrypted.
    scale : float
        Scaling factor.

    Returns
    -------
    numpy.ndarray
        Ciphertext of the encoded plaintext of the floating-point data.

    See Also
    --------
    eclib.regev.encrypt
    eclib.regev.encode
    """

    return encrypt(params, pk, encode(params, x, scale))


def dec(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_], scale: float
) -> ArrayLike:
    """
    Decrypts and decodes a scalar, vector, or matrix ciphertext `c` using a secret key
    `sk`.

    Parameters
    ----------
    params : eclib.regev.PublicParameters
        Cryptosystem parameters.
    sk : eclib.regev.SecretKey
        Secret key used for decryption.
    c : numpy.ndarray
        Ciphertext to be decrypted and decoded.
    scale : float
        Scaling factor.

    Returns
    -------
    array_like
        Decoded floating-point data of the decrypted plaintext.

    See Also
    --------
    eclib.regev.decrypt
    eclib.regev.decode
    """

    return decode(params, decrypt(params, sk, c), scale)


def _encrypt(params: PublicParameters, pk: PublicKey, m: int) -> NDArray[np.object_]:
    """
    Encrypts a message `m` using a public key `pk`.

    Parameters
    ----------
    params : eclib.regev.PublicParameters
        Cryptosystem parameters.
    pk : eclib.regev.PublicKey
        Public key used for encryption.
    m : int
        Plaintext to be encrypted.

    Returns
    -------
    numpy.ndarray
        Ciphertext of the plaintext.
    """

    r = np.array([[ru.get_rand(0, 2)] for _ in range(params.m)], dtype=object)

    return (
        pk.B @ r
        + floor(params.q / params.t)
        * m
        * np.block([[1], [np.zeros([params.n, 1], dtype=object)]])
    ) % params.q


def _decrypt(params: PublicParameters, sk: SecretKey, c: NDArray[np.object_]) -> int:
    """
    Decrypts a ciphertext `c` using a secret key `sk`.

    Parameters
    ----------
    params : eclib.regev.PublicParameters
        Cryptosystem parameters.
    sk : eclib.regev.SecretKey
        Secret key used for decryption.
    c : numpy.ndarray
        Ciphertext to be decrypted.

    Returns
    -------
    int
        Decrypted plaintext.
    """

    return (
        floor(
            (params.t / params.q) * ((np.block([1, -sk.s.T]) @ c).item() % params.q)
            + 0.5
        )
        % params.t
    )


def _add(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    Computes a ciphertext of the addition of two plaintexts corresponding to
    ciphertexts `c1` and `c2`.

    Parameters
    ----------
    params : eclib.regev.PublicParameters
        Cryptosystem parameters.
    c1 : numpy.ndarray
        Ciphertext of the first plaintext.
    c2 : numpy.ndarray
        Ciphertext of the second plaintext.

    Returns
    -------
    numpy.ndarray
        Ciphertext of the addition of the plaintexts.
    """

    return (c1 + c2) % params.q


def _int_mult(
    params: PublicParameters, m: int, c: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    Computes a ciphertext of the product of a plaintext `m` and another plaintext
    corresponding to a ciphertext `c`.

    Parameters
    ----------
    params : eclib.regev.PublicParameters
        Cryptosystem parameters.
    m : int
        Plaintext to be multiplied.
    c : numpy.ndarray
        Ciphertext of a plaintext to be multiplied.

    Returns
    -------
    numpy.ndarray
        Ciphertext of the product of the plaintexts.
    """

    return (m * c) % params.q


def _encode(params: PublicParameters, x: float, scale: float) -> int:
    """
    Encodes a floating-point number `x` into a plaintext.

    Parameters
    ----------
    params : eclib.regev.PublicParameters
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

    if m < -((params.t - 1) // 2):
        raise ValueError("Underflow")

    elif m > (params.t // 2):
        raise ValueError("Overflow")

    else:
        return m % params.t


def _decode(params: PublicParameters, m: int, scale: float) -> float:
    """
    Decodes a plaintext `m` into a floating-point number.

    Parameters
    ----------
    params : eclib.regev.PublicParameters
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

    return (m - floor(m / params.t + 0.5) * params.t) * scale
