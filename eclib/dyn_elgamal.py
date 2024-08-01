#! /usr/bin/env python3

"""
dyn_elgamal.py

This module implements the dynamic-key ElGamal encryption scheme. The dynamic-key
ElGamal encryption scheme is a variant of the ElGamal encryption scheme that allows
for updating keys and ciphertexts. This is useful for applications where the key pair
needs to be updated periodically to enhance security. The module provides
functionalities for generating public parameters, public and secret keys, encryption,
decryption, and homomorphic operations (multiplication). It also includes functions
for encoding and decoding floating-point data into and from plaintexts.

Classes:
    Token: Represents a token used for updating keys and ciphertexts.

Functions:
    keygen: Generates public parameters, a public key, and a secret key.
    encrypt: Encrypts a scalar, vector, or matrix plaintext.
    decrypt: Decrypts a scalar, vector, or matrix ciphertext.
    mult: Computes a ciphertext of the Hadamard product of two scalar, vector, or matrix
        plaintexts.
    encode: Encodes a scalar, vector, or matrix floating-point data into a plaintext.
    decode: Decodes a scalar, vector, or matrix plaintext into floating-point data.
    enc: Encrypts and encodes a scalar, vector, or matrix floating-point data.
    dec: Decrypts and decodes a scalar, vector, or matrix ciphertext.
    dec_add: Decrypts and computes the sum of row-wise elements of a ciphertext.
    update_key: Updates a public key and secret key with a token.
    update_ct: Updates a ciphertext using a token.

Dependencies:
    numpy: Fundamental package for scientific computing with Python.
    numpy.typing: Type hints for NumPy.
    eclib.randutils: Utility functions for generating random numbers.
    eclib.elgamal: ElGamal encryption scheme.
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

    Attributes:
        s (int): Previous secret key value.
        h (int): Previous public key value.
    """

    s: int
    h: int


def keygen(bit_length: int) -> tuple[PublicParameters, PublicKey, SecretKey]:
    """
    This function is the same as :func:`~elgamal.keygen`.
    """

    return elgamal.keygen(bit_length)


def encrypt(
    params: PublicParameters, pk: PublicKey, m: ArrayLike
) -> NDArray[np.object_]:
    """
    This function is the same as :func:`~elgamal.encrypt`.
    """

    return elgamal.encrypt(params, pk, m)


def decrypt(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_]
) -> ArrayLike:
    """
    This function is the same as :func:`~elgamal.decrypt`.
    """

    return elgamal.decrypt(params, sk, c)


def mult(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    This function is the same as :func:`~elgamal.mult`.
    """

    return elgamal.mult(params, c1, c2)


def encode(params: PublicParameters, x: ArrayLike, delta: float) -> ArrayLike:
    """
    This function is the same as :func:`~elgamal.encode`.
    """

    return elgamal.encode(params, x, delta)


def decode(params: PublicParameters, m: ArrayLike, delta: float) -> ArrayLike:
    """
    This function is the same as :func:`~elgamal.decode`.
    """

    return elgamal.decode(params, m, delta)


def enc(
    params: PublicParameters, pk: PublicKey, x: ArrayLike, delta: float
) -> NDArray[np.object_]:
    """
    This function is the same as :func:`~elgamal.enc`.
    """

    return elgamal.enc(params, pk, x, delta)


def dec(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_], delta: float
) -> ArrayLike:
    """
    This function is the same as :func:`~elgamal.dec`.
    """

    return elgamal.dec(params, sk, c, delta)


def dec_add(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_], delta: float
) -> ArrayLike:
    """
    This function is the same as :func:`~elgamal.dec_add`.
    """

    return elgamal.dec_add(params, sk, c, delta)


def update_key(
    params: PublicParameters, pk: PublicKey, sk: SecretKey
) -> tuple[PublicKey, SecretKey, Token]:
    """
    Updates a public key `pk` and secret key `sk` with a token used for updating
    ciphertexts.

    Args:
        params (PublicParameters): Crptosystem parameters.
        pk (PublicKey): Public key to be updated.
        sk (SecretKey): Secret key to be updated.

    Returns:
        tuple[PublicKey, SecretKey, Token]: Tuple containing the updated public key,
        secret key, and token.

    See Also:
        update_ct
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

    Args:
        params (PublicParameters): Cryptosystem parameters.
        c (NDArray[np.object_]): Ciphertext to be updated.
        t (Token): Token used for updating the ciphertext.

    Returns:
        NDArray[np.object_]: Updated ciphertext.

    Raises:
        ValueError: If the ciphertext is not a scalar, vector, or matrix.

    See Also:
        update_key
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

    Args:
        params (PublicParameters): Cryptosystem parameters.
        c (NDArray[np.object_]): Ciphertext to be updated.
        t (Token): Token used for updating the ciphertext.

    Returns:
        NDArray[np.object_]: Updated ciphertext.
    """

    c = np.asarray(c, dtype=object)
    c_updated = np.zeros_like(c)

    r = ru.get_rand(1, params.q)
    c_updated[0] = (c[0] * pow(params.g, r, params.p)) % params.p
    c_updated[1] = (
        pow(c_updated[0], t.s, params.p) * c[1] * pow(t.h, r, params.p)
    ) % params.p

    return c_updated
