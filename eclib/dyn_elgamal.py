#! /usr/bin/env python3

"""Dynamic-key ElGamal encryption scheme.

This module implements the dynamic-key ElGamal encryption scheme, a variant of
the ElGamal encryption scheme, that allows for updating keys and ciphertexts. This is
useful for applications where the key pair needs to be updated periodically to enhance
security. The module provides functionalities for generating public and secret keys,
encryption, decryption, and homomorphic operations (multiplication). It also includes
functions for encoding and decoding floating-point data into and from plaintexts.

Classes
-------
- Token

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
- update_key
- update_ct
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

import eclib.randutils as ru
from eclib import elgamal
from eclib.elgamal import PublicKey, PublicParameters, SecretKey


@dataclass(slots=True)
class Token:
    """
    Represents a token used for updating keys and ciphertexts in the dynamic-key
    ElGamal encryption scheme.

    Attributes
    ----------
    s : int
        Previous secret key value.
    h : int
        Previous public key value.
    """

    s: int
    h: int


def keygen(bit_length: int) -> tuple[PublicParameters, PublicKey, SecretKey]:
    """
    This function is the same as :func:`eclib.elgamal.keygen`.
    """

    return elgamal.keygen(bit_length)


def encrypt(
    params: PublicParameters, pk: PublicKey, m: ArrayLike
) -> NDArray[np.object_]:
    """
    This function is the same as :func:`eclib.elgamal.encrypt`.
    """

    return elgamal.encrypt(params, pk, m)


def decrypt(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_]
) -> ArrayLike:
    """
    This function is the same as :func:`eclib.elgamal.decrypt`.
    """

    return elgamal.decrypt(params, sk, c)


def mult(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    This function is the same as :func:`eclib.elgamal.mult`.
    """

    return elgamal.mult(params, c1, c2)


def encode(params: PublicParameters, x: ArrayLike, scale: float) -> ArrayLike:
    """
    This function is the same as :func:`eclib.elgamal.encode`.
    """

    return elgamal.encode(params, x, scale)


def decode(params: PublicParameters, m: ArrayLike, scale: float) -> ArrayLike:
    """
    This function is the same as :func:`eclib.elgamal.decode`.
    """

    return elgamal.decode(params, m, scale)


def enc(
    params: PublicParameters, pk: PublicKey, x: ArrayLike, scale: float
) -> NDArray[np.object_]:
    """
    This function is the same as :func:`eclib.elgamal.enc`.
    """

    return elgamal.enc(params, pk, x, scale)


def dec(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_], scale: float
) -> ArrayLike:
    """
    This function is the same as :func:`eclib.elgamal.dec`.
    """

    return elgamal.dec(params, sk, c, scale)


def dec_add(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_], scale: float
) -> ArrayLike:
    """
    This function is the same as :func:`eclib.elgamal.dec_add`.
    """

    return elgamal.dec_add(params, sk, c, scale)


def update_key(
    params: PublicParameters, pk: PublicKey, sk: SecretKey
) -> tuple[PublicKey, SecretKey, Token]:
    """
    Updates a public key `pk` and secret key `sk` with a token used for updating
    ciphertexts.

    Parameters
    ----------
    params : eclib.elgamal.PublicParameters
        Cryptosystem parameters.
    pk : eclib.elgamal.PublicKey
        Public key to be updated.
    sk : eclib.elgamal.SecretKey
        Secret key to be updated.

    Returns
    -------
    pk_updated : eclib.elgamal.PublicKey
        Updated public key.
    sk_updated : eclib.elgamal.SecretKey
        Updated secret key.
    t : eclib.dyn_elgamal.Token
        Token used for updating ciphertexts.

    See Also
    --------
    eclib.dyn_elgamal.update_ct
    """

    sk_updated = SecretKey(params=None)
    pk_updated = PublicKey(params=None, sk=None)

    t = Token(ru.get_rand(1, params.q), pk.h)
    sk_updated.s = (sk.s + t.s) % params.q
    pk_updated.h = pk.h * pow(params.g, t.s, params.p) % params.p

    return pk_updated, sk_updated, t


def update_ct(
    params: PublicParameters, c: NDArray[np.object_], t: Token
) -> NDArray[np.object_]:
    """
    Updates a scalar, vector, or matrix ciphertext `c` using a token `t`.

    Parameters
    ----------
    params : eclib.elgamal.PublicParameters
        Cryptosystem parameters.
    c : numpy.ndarray
        Ciphertext to be updated.
    t : eclib.dyn_elgamal.Token
        Token used for updating the ciphertext.

    Returns
    -------
    numpy.ndarray
        Updated ciphertext.

    Raises
    ------
    ValueError
        If the ciphertext is not a scalar, vector, or matrix.

    See Also
    --------
    eclib.dyn_elgamal.update_key
    """

    c = np.asarray(c, dtype=object)

    match c.ndim - 1:
        case 0:
            return _update_ct(params, c, t)

        case 1:
            return np.array(
                [_update_ct(params, c[i], t) for i in range(c.shape[0])],
                dtype=object,
            )

        case 2:
            return np.array(
                [
                    [_update_ct(params, c[i][j], t) for j in range(c.shape[1])]
                    for i in range(c.shape[0])
                ],
                dtype=object,
            )

        case _:
            raise ValueError


def _update_ct(
    params: PublicParameters, c: NDArray[np.object_], t: Token
) -> NDArray[np.object_]:
    """
    Updates a ciphertext `c` using a token `t`.

    Parameters
    ----------
    params : eclib.elgamal.PublicParameters
        Cryptosystem parameters.
    c : numpy.ndarray
        Ciphertext to be updated.
    t : eclib.dyn_elgamal.Token
        Token used for updating the ciphertext.

    Returns
    -------
    numpy.ndarray
        Updated ciphertext.
    """

    c = np.asarray(c, dtype=object)
    c_updated = np.zeros_like(c)

    r = ru.get_rand(1, params.q)
    c_updated[0] = (c[0] * pow(params.g, r, params.p)) % params.p
    c_updated[1] = (
        pow(c_updated[0], t.s, params.p) * c[1] * pow(t.h, r, params.p)
    ) % params.p

    return c_updated
