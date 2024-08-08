#! /usr/bin/env python3

"""ElGamal encryption scheme.

This module implements the ElGamal encryption scheme, which is a public-key
cryptosystem based on the Decisional Diffie-Hellman (DDH) assumption. It provides
functionalities for generating public and secret keys, encryption, decryption, and
homomorphic operations (multiplication). It also includes functions for encoding and
decoding floating-point data into and from plaintexts.

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
- mult
- encode
- decode
- enc
- dec
- dec_add
"""

from dataclasses import dataclass
from math import floor
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

import eclib.primeutils as pu
import eclib.randutils as ru


@dataclass(slots=True)
class PublicParameters:
    """
    Represents public parameters of the ElGamal encryption scheme.

    Attributes
    ----------
    p : int
        Prime number representing the modulus of a cyclic group used as a plaintext
        space.
    q : int
        Prime number representing the order of the cyclic group.
    g : int
        Generator of the cyclic group.
    """

    p: int
    q: int
    g: int

    def __init__(self, bit_length: Optional[int] = None):
        """
        Initializes a new PublicParameters object.

        Parameters
        ----------
        bit_length : int, optional
            Desired bit length for the prime number `q`.

        Returns
        -------
        None

        Note
        ----
        If `bit_length` is None, the public parameters will be initialized with `0`.
        Otherwise, the public parameters will be initialized with a Sophie Germain
        prime `q` and a safe prime `p` such that `p = 2q + 1`, and a generator `g` of
        the cyclic group of order `q` modulo `p`.
        """

        if bit_length is None:
            self.p = self.q = self.g = 0

        else:
            self.q, self.p = pu.get_safe_prime(bit_length)
            self.g = _get_generator(self.q, self.p)


@dataclass(slots=True)
class SecretKey:
    """
    Represents a secret key of the ElGamal encryption scheme.

    Attributes
    ----------
    s : int
        Secret key value.
    """

    s: int

    def __init__(self, params: Optional[PublicParameters] = None):
        """
        Initializes a new SecretKey object.

        Parameters
        ----------
        params : eclib.elgamal.PublicParameters, optional
            Cryptosystem parameters used for computing the secret key.

        Returns
        -------
        None

        Note
        ----
        If `params` is None, the secret key will be initialized with `0`. Otherwise, it
        will be initialized as a random number in `[1, q - 1]`, where `q` is a prime
        number representing the order of a cyclic group used as a plaintext space.

        See Also
        --------
        eclib.elgamal.PublicParameters
        """

        if params is None:
            self.s = 0

        else:
            self.s = ru.get_rand(1, params.q)


@dataclass(slots=True)
class PublicKey:
    """
    Represents a public key of the ElGamal encryption scheme.

    Attributes
    ----------
    h : int
        Public key value.
    """

    h: int

    def __init__(
        self, params: Optional[PublicParameters] = None, sk: Optional[SecretKey] = None
    ):
        """
        Initializes a new PublicKey object.

        Parameters
        ----------
        params : eclib.elgamal.PublicParameters, optional
            Cryptosystem parameters used for computing the public key.
        sk : eclib.elgamal.SecretKey, optional
            Secret key used for computing the public key.

        Returns
        -------
        None

        Note
        ----
        If `params` or `sk` is None, the public key will be initialized with `0`.
        Otherwise, it will be computed as `h = g^s mod p`, where `g` is a generator of
        a cyclic group used as a plaintext space, `s` is a secret key value, and `p` is
        a prime number representing the modulus of the cyclic group.

        See Also
        --------
        eclib.elgamal.PublicParameters
        eclib.elgamal.SecretKey
        """

        if params is None or sk is None:
            self.h = 0

        else:
            self.h = pow(params.g, sk.s, params.p)


def keygen(bit_length: int) -> tuple[PublicParameters, PublicKey, SecretKey]:
    """
    Generates public parameters, a public key, and a secret key.

    Parameters
    ----------
    bit_length : int
        Desired bit length of a prime number representing the order of a cyclic group
        used as a plaintext space.

    Returns
    -------
    params : eclib.elgamal.PublicParameters
        Cryptosystem parameters.
    pk : eclib.elgamal.PublicKey
        Public key used for encryption.
    sk : eclib.elgamal.SecretKey
        Secret key used for decryption.

    See Also
    --------
    eclib.elgamal.PublicParameters
    eclib.elgamal.PublicKey
    eclib.elgamal.SecretKey
    """

    params = PublicParameters(bit_length)

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
    params : eclib.elgamal.PublicParameters
        Cryptosystem parameters.
    pk : eclib.elgamal.PublicKey
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
    eclib.elgamal.decrypt
    eclib.elgamal.enc
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
    params : eclib.elgamal.PublicParameters
        Cryptosystem parameters.
    sk : eclib.elgamal.SecretKey
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
    eclib.elgamal.encrypt
    eclib.elgamal.dec
    """

    c = np.asarray(c, dtype=object)

    match c.ndim - 1:
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


def mult(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    Computes a ciphertext of the Hadamard product of two scalar, vector, or matrix
    plaintexts corresponding to ciphertexts `c1` and `c2`.

    Parameters
    ----------
    params : eclib.elgamal.PublicParameters
        Cryptosystem parameters.
    c1 : numpy.ndarray
        Ciphertext of the first plaintext.
    c2 : numpy.ndarray
        Ciphertext of the second plaintext.

    Returns
    -------
    numpy.ndarray
        Ciphertext of the Hadamard product of the plaintexts.

    Raises
    ------
    ValueError
        If the ciphertexts are not the following types of appropriate sizes:
        scalar-scalar, scalar-vector, scalar-matrix, vector-vector, matrix-vector, or
        matrix-matrix.
    """

    c1 = np.asarray(c1, dtype=object)
    c2 = np.asarray(c2, dtype=object)

    match c1.ndim - 1:
        case 0 if c2.ndim - 1 == 0:
            return _mult(params, c1, c2)

        case 0 if c2.ndim - 1 == 1:
            return np.array(
                [_mult(params, c1, c2[i]) for i in range(c2.shape[0])],
                dtype=object,
            )

        case 0 if c2.ndim - 1 == 2:
            return np.array(
                [
                    [_mult(params, c1, c2[i][j]) for j in range(c2.shape[1])]
                    for i in range(c2.shape[0])
                ],
                dtype=object,
            )

        case 1 if c1.shape == c2.shape:
            return np.array(
                [_mult(params, c1[i], c2[i]) for i in range(c1.shape[0])],
                dtype=object,
            )

        case 2 if c2.ndim - 1 == 1 and c1.shape[1] == c2.shape[0]:
            return np.array(
                [
                    [_mult(params, c1[i][j], c2[j]) for j in range(c1.shape[1])]
                    for i in range(c1.shape[0])
                ],
                dtype=object,
            )

        case 2 if c1.shape == c2.shape:
            return np.array(
                [
                    [_mult(params, c1[i][j], c2[i][j]) for j in range(c1.shape[1])]
                    for i in range(c1.shape[0])
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
    params : eclib.elgamal.PublicParameters
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
    eclib.elgamal.decode
    eclib.elgamal.enc
    """

    f = np.frompyfunc(_encode, 3, 1)
    return f(params, x, scale)


def decode(params: PublicParameters, m: ArrayLike, scale: float) -> ArrayLike:
    """
    Decodes a scalar, vector, or matrix plaintext `m` into a floating-point data.

    Parameters
    ----------
    params : eclib.elgamal.PublicParameters
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
    eclib.elgamal.encode
    eclib.elgamal.dec
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
    params : eclib.elgamal.PublicParameters
        Cryptosystem parameters.
    pk : eclib.elgamal.PublicKey
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
    eclib.elgamal.encrypt
    eclib.elgamal.encode
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
    params : eclib.elgamal.PublicParameters
        Cryptosystem parameters.
    sk : eclib.elgamal.SecretKey
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
    eclib.elgamal.decrypt
    eclib.elgamal.decode
    """

    return decode(params, decrypt(params, sk, c), scale)


def dec_add(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_], scale: float
) -> ArrayLike:
    """
    Decrypts and computes the sum of row-wise elements of a scalar, vector, or matrix
    ciphertext `c` using a secret key `sk`.

    Parameters
    ----------
    params : eclib.elgamal.PublicParameters
        Cryptosystem parameters.
    sk : eclib.elgamal.SecretKey
        Secret key used for decryption.
    c : numpy.ndarray
        Ciphertext to be decrypted and summed.
    scale : float
        Scaling factor.

    Returns
    -------
    array_like
        Decoded floating-point data of the sum of the row-wise elements of the
        decrypted plaintext.

    Raises
    ------
    ValueError
        If the ciphertext is not a scalar, vector, or matrix.

    See Also
    --------
    eclib.elgamal.dec
    """

    c = np.asarray(c, dtype=object)

    match c.ndim - 1:
        case 0:
            return dec(params, sk, c, scale)

        case 1:
            return np.sum(dec(params, sk, c, scale), axis=0)

        case 2:
            return np.sum(dec(params, sk, c, scale), axis=1)

        case _:
            raise ValueError


def _encrypt(params: PublicParameters, pk: PublicKey, m: int) -> NDArray[np.object_]:
    """
    Encrypts a plaintext `m` using a public key `pk`.

    Parameters
    ----------
    params : eclib.elgamal.PublicParameters
        Cryptosystem parameters.
    pk : eclib.elgamal.PublicKey
        Public key used for encryption.
    m : int
        Plaintext to be encrypted.

    Returns
    -------
    numpy.ndarray
        Ciphertext of the plaintext.
    """

    r = ru.get_rand(1, params.q)

    return np.array(
        [pow(params.g, r, params.p), (m * pow(pk.h, r, params.p)) % params.p],
        dtype=object,
    )


def _decrypt(params: PublicParameters, sk: SecretKey, c: NDArray[np.object_]) -> int:
    """
    Decrypts a ciphertext `c` using a secret key `sk`.

    Parameters
    ----------
    params : eclib.elgamal.PublicParameters
        Cryptosystem parameters.
    sk : eclib.elgamal.SecretKey
        Secret key used for decryption.
    c : numpy.ndarray
        Ciphertext to be decrypted.

    Returns
    -------
    int
        Decrypted plaintext.
    """

    return (pow(c[0], -sk.s, params.p) * c[1]) % params.p


def _mult(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    Computes a ciphertext of the product of two plaintexts corresponding to ciphertexts
    `c1` and `c2`.

    Parameters
    ----------
    params : eclib.elgamal.PublicParameters
        Cryptosystem parameters.
    c1 : numpy.ndarray
        Ciphertext of the first plaintext.
    c2 : numpy.ndarray
        Ciphertext of the second plaintext.

    Returns
    -------
    numpy.ndarray
        Ciphertext of the product of the plaintexts.
    """

    return np.array(
        [(c1[0] * c2[0]) % params.p, (c1[1] * c2[1]) % params.p], dtype=object
    )


def _encode(params: PublicParameters, x: float, scale: float) -> int:
    """
    Encodes a floating-point number `x` into a plaintext.

    Parameters
    ----------
    params : eclib.elgamal.PublicParameters
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
    first_decimal_place = (x / scale * 10) % 10

    if m < 0:
        if m < -params.q:
            raise ValueError("Underflow")

        else:
            m += params.p

    elif m > params.q:
        raise ValueError("Overflow")

    if x / scale == int(x / scale) or first_decimal_place >= 5:
        for i in range(params.q):
            if m - i > 0 and _is_element(m - i, params.q, params.p):
                return m - i

            elif m + i < params.p and _is_element(m + i, params.q, params.p):
                return m + i

    else:
        for i in range(params.q):
            if m + i < params.p and _is_element(m + i, params.q, params.p):
                return m + i

            elif m - i > 0 and _is_element(m - i, params.q, params.p):
                return m - i

    raise ValueError


def _decode(params: PublicParameters, m: int, scale: float) -> float:
    """
    Decodes a plaintext `m` into a floating-point number.

    Parameters
    ----------
    params : eclib.elgamal.PublicParameters
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

    if m > params.q:
        return (m - params.p) * scale

    else:
        return m * scale


def _is_generator(g: int, q: int, p: int) -> bool:
    """
    Checks if an integer `g` is a generator of a cyclic group of order `q` modulo `p`.

    Parameters
    ----------
    g : int
        Integer to be checked.
    q : int
        Prime number representing the order of the cyclic group.
    p : int
        Prime number representing the modulus of the cyclic group.

    Returns
    -------
    bool
        True if `g` is a generator of the cyclic group, False otherwise.
    """

    if g <= 1 or g >= p:
        return False

    elif pow(g, q, p) == 1:
        return True

    else:
        return False


def _is_element(m: int, q: int, p: int) -> bool:
    """
    Check if an integer `m` is an element of a cyclic group of order `q` modulo `p`.

    Parameters
    ----------
    m : int
        Integer to be checked.
    q : int
        Prime number representing the order of the cyclic group.
    p : int
        Prime number representing the modulus of the cyclic group.

    Returns
    -------
    bool
        True if `m` is an element of the cyclic group, False otherwise.
    """

    if m <= 0 or m >= p:
        return False

    elif pow(m, q, p) == 1:
        return True

    else:
        return False


def _get_generator(q: int, p: int) -> int:
    """
    Returns a generator for a cyclic group of order `q` modulo `p`.

    Parameters
    ----------
    p : int
        Prime number representing the order of the cyclic group.
    q : int
        Prime number representing the modulus of the cyclic group.

    Returns:
        int: Generator for the cyclib group.
    """

    g = 2
    while _is_generator(g, q, p) is False:
        g += 1

    return g
