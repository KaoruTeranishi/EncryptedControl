#! /usr/bin/env python3

"""
gsw_lwe.py

This module implements the GSW-LWE encryption scheme, which is a fully homomorphic
encryption scheme that combines the GSW cryptosystem and the Regev (LWE) cryptosystem.
The GSW-LWE encryption scheme supports the outer product of a GSW ciphertext and an LWE
ciphertext: GSW x LWE -> LWE. This allows for efficient homomorphic multiplication. The
module provides functionalities for generating public parameters, public and secret
keys, encryption, decryption, and homomorphic operations (addition, multiplication, and
integer multiplication). It also includes functions for encoding and decoding floating-
point data into and from plaintexts.

Classes:
    PublicParameters: Represents public parameters of the GSW-LWE encryption scheme.

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
    eclib.gsw: GSW encryption scheme.
    eclib.regev: Regev (LWE) encryption scheme.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eclib import gsw, regev
from eclib.regev import PublicKey, SecretKey


@dataclass(slots=True)
class PublicParameters:
    """
    Represents public parameters of the GSW-LWE encryption scheme.

    Attributes:
        lwe_params (regev.PublicParameters): Regev (LWE) cryptosystem parameters.
        gsw_params (gsw.PublicParameters): GSW cryptosystem parameters.
    """

    lwe_params: regev.PublicParameters
    gsw_params: gsw.PublicParameters

    def __init__(self, n: int, t: int, q: int, sigma: float, m: Optional[int] = None):
        """
        Initializes a new PublicParameters object.

        Args:
            n (int): Dimension of a lattice, which is equal to the dimension of secret
                key.
            t (int): Modulus of a plaintext space.
            q (int): Modulus of a ciphertext space.
            sigma (float): Standard deviation of the discrete Gaussian distribution
                with mean zero used as an error distribution.
            m (int, optional, default = None): Subdimension of the lattice.

        Note:
            If `m` is not provided, it is set to `2 * n * ceil(log2(q))`.
        """
        self.lwe_params = regev.PublicParameters(n, t, q, sigma, m)
        self.gsw_params = gsw.PublicParameters(n, q, sigma, m)


def keygen(n: int, t: int, q: int, sigma: float, m: Optional[int] = None):
    """
    Generates public parameters, a public key, and a secret key.

    Args:
        n (int): Dimension of a lattice, which is equal to the dimension of secret key.
        t (int): Modulus of a plaintext space.
        q (int): Modulus of a ciphertext space.
        sigma (float): Standard deviation of the discrete Gaussian distribution with
            mean zero used as an error distribution.
        m (int, optional, default = None): Subdimension of the lattice.

    Returns:
        tuple[PublicParameters, PublicKey, SecretKey]: Tuple containing the public
            parameters and Regev (LWE) public and secret keys. The public parameters
            consist of Regev (LWE) and GSW public parameters.

    Note:
        If `m` is not provided, it is set to `2 * n * ceil(log2(q))`.

    See Also:
        PublicParameters
        :class:`~gsw.PublicParameters`
        :class:`~regev.PublicParameters`
        :class:`~regev.PublicKey`
        :class:`~regev.SecretKey`
    """

    params = PublicParameters(n, t, q, sigma, m)

    sk = SecretKey(params.lwe_params)

    pk = PublicKey(params.lwe_params, sk)

    return params, pk, sk


def encrypt(
    params: PublicParameters, pk: PublicKey, m: ArrayLike
) -> NDArray[np.object_]:
    """
    This function is the same as :func:`~regev.encrypt`.
    """

    return regev.encrypt(params.lwe_params, pk, m)


def encrypt_gsw(
    params: PublicParameters, pk: PublicKey, m: ArrayLike
) -> NDArray[np.object_]:
    """
    This function is the same as :func:`~gsw.encrypt`.
    """

    return gsw.encrypt(params.gsw_params, pk, m)


def decrypt(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_]
) -> ArrayLike:
    """
    This function is the same as :func:`~regev.decrypt`.
    """

    return regev.decrypt(params.lwe_params, sk, c)


def decrypt_gsw(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_]
) -> ArrayLike:
    """
    This function is the same as :func:`~gsw.decrypt`.
    """

    return gsw.decrypt(params.gsw_params, sk, c)


def add(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    This function is the same as :func:`~regev.add`.
    """

    return regev.add(params.lwe_params, c1, c2)


def elementwise_add(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    This function is the same as :func:`~regev.elementwise_add`.
    """

    return regev.elementwise_add(params.lwe_params, c1, c2)


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
        case 0:
            return gsw.mult(params.gsw_params, c1, c2)

        case 1 if c2.ndim - 2 == 1 and c1.shape[0] == c2.shape[0]:
            c_s = np.zeros([params.lwe_params.n + 1, 1], dtype=object)

            for i in range(c1.shape[0]):
                c_s = regev.add(
                    params.lwe_params, c_s, gsw.mult(params.gsw_params, c1[i], c2[i])
                )

            return c_s

        case 2 if c2.ndim - 2 == 1 and c1.shape[1] == c2.shape[0]:
            c_v = np.zeros([c1.shape[0], params.lwe_params.n + 1, 1], dtype=object)

            for i in range(c1.shape[0]):
                for j in range(c1.shape[1]):
                    c_v[i] = regev.add(
                        params.lwe_params,
                        c_v[i],
                        gsw.mult(params.gsw_params, c1[i][j], c2[j]),
                    )

            return c_v

        case 2 if c2.ndim - 2 == 2 and c1.shape[1] == c2.shape[0]:
            c_m = np.zeros(
                [c1.shape[0], c2.shape[1], params.lwe_params.n + 1, 1], dtype=object
            )

            for i in range(c1.shape[0]):
                for j in range(c2.shape[1]):
                    for k in range(c1.shape[1]):
                        c_m[i][j] = regev.add(
                            params.lwe_params,
                            c_m[i][j],
                            gsw.mult(params.gsw_params, c1[i][k], c2[k][j]),
                        )

            return c_m

        case _:
            raise ValueError


def elementwise_mult(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    This function is the same as :func:`~gsw.elementwise_mult`.
    """

    return gsw.elementwise_mult(params.gsw_params, c1, c2)


def int_mult(
    params: PublicParameters, m: ArrayLike, c: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    This function is the same as :func:`~regev.int_mult`.
    """

    return regev.int_mult(params.lwe_params, m, c)


def elementwise_int_mult(
    params: PublicParameters, m: ArrayLike, c: NDArray[np.object_]
) -> NDArray[np.object_]:
    """
    This function is the same as :func:`~regev.elementwise_int_mult`.
    """

    return regev.elementwise_int_mult(params.lwe_params, m, c)


def encode(params: PublicParameters, x: ArrayLike, delta: float) -> ArrayLike:
    """
    This function is the same as :func:`~regev.encode`.
    """

    return regev.encode(params.lwe_params, x, delta)


def decode(params: PublicParameters, m: ArrayLike, delta: float) -> ArrayLike:
    """
    This function is the same as :func:`~regev.decode`.
    """

    return regev.decode(params.lwe_params, m, delta)


def enc(
    params: PublicParameters, pk: PublicKey, x: ArrayLike, delta: float
) -> NDArray[np.object_]:
    """
    This function is the same as :func:`~regev.enc`.
    """

    return regev.enc(params.lwe_params, pk, x, delta)


def enc_gsw(
    params: PublicParameters, pk: PublicKey, x: ArrayLike, delta: float
) -> NDArray[np.object_]:
    """
    This function is the same as :func:`~gsw.enc`.
    """

    return gsw.enc(params.gsw_params, pk, x, delta)


def dec(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_], delta: float
) -> ArrayLike:
    """
    This function is the same as :func:`~regev.dec`.
    """

    return regev.dec(params.lwe_params, sk, c, delta)


def dec_gsw(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_], delta: float
) -> ArrayLike:
    """
    This function is the same as :func:`~gsw.dec`.
    """

    return gsw.dec(params.gsw_params, sk, c, delta)
