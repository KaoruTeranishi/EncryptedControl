#! /usr/bin/env python3

from dataclasses import dataclass
from math import floor, gcd

import numpy as np
from numpy.typing import ArrayLike, NDArray

import eclib.modutils as mu
import eclib.primeutils as pu
import eclib.randutils as ru
from eclib import exceptions


@dataclass(slots=True)
class SecretKey:
    p: int
    q: int
    lmd: int
    mu: int

    def __init__(self, bit_length: int):
        self.p, self.q = pu.get_semiprime_factors(bit_length)
        self.lmd = (self.p - 1) * (self.q - 1)
        self.mu = pow(self.lmd, -1, self.p * self.q)


@dataclass(slots=True)
class PublicParameters:
    n: int
    n_square: int

    def __init__(self, sk: SecretKey):
        self.n = sk.p * sk.q
        self.n_square = self.n**2


@dataclass(slots=True)
class PublicKey:
    g: int

    def __init__(self, params: PublicParameters):
        self.g = params.n + 1


def keygen(bit_length: int) -> tuple[PublicParameters, PublicKey, SecretKey]:
    sk = SecretKey(bit_length)

    params = PublicParameters(sk)

    pk = PublicKey(params)

    return params, pk, sk


def encrypt(
    params: PublicParameters, pk: PublicKey, m: ArrayLike
) -> int | NDArray[np.object_]:
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
            raise exceptions.EncryptionError


def decrypt(
    params: PublicParameters, sk: SecretKey, c: int | NDArray[np.object_]
) -> ArrayLike:
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
            raise exceptions.DecryptionError


def add(
    params: PublicParameters,
    c1: int | NDArray[np.object_],
    c2: int | NDArray[np.object_],
) -> int | NDArray[np.object_]:
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
                raise exceptions.HomomorphicOperationError

    else:
        raise exceptions.HomomorphicOperationError


def elementwise_add(
    params: PublicParameters,
    c1: int | NDArray[np.object_],
    c2: int | NDArray[np.object_],
) -> int | NDArray[np.object_]:
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
        raise exceptions.HomomorphicOperationError


def int_mult(
    params: PublicParameters, m: ArrayLike, c: int | NDArray[np.object_]
) -> int | NDArray[np.object_]:
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

        case 1 if m.shape == c.shape:
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
            raise exceptions.HomomorphicOperationError


def elementwise_int_mult(
    params: PublicParameters, m: ArrayLike, c: int | NDArray[np.object_]
) -> int | NDArray[np.object_]:
    m = np.asarray(m, dtype=object)
    c = np.asarray(c, dtype=object)

    match m.ndim:
        case 0:
            return int_mult(params, m.item(), c.item())

        case 1 if m.shape == c.shape:
            return np.array(
                [_int_mult(params, m[i], c[i]) for i in range(m.shape[0])],
                dtype=object,
            )

        case 2 if c.ndim == 1 and m.shape[1] == c.shape[0]:
            return np.array(
                [
                    [_int_mult(params, m[i][j], c[j]) for j in range(m.shape[1])]
                    for i in range(c.shape[0])
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
            raise exceptions.HomomorphicOperationError


def encode(params: PublicParameters, x: ArrayLike, delta: float) -> ArrayLike:
    f = np.frompyfunc(_encode, 3, 1)
    return f(params, x, delta)


def decode(params: PublicParameters, m: ArrayLike, delta: float) -> ArrayLike:
    f = np.frompyfunc(_decode, 3, 1)
    return f(params, m, delta)


def enc(
    params: PublicParameters, pk: PublicKey, x: ArrayLike, delta: float
) -> int | NDArray[np.object_]:
    return encrypt(params, pk, encode(params, x, delta))


def dec(
    params: PublicParameters, sk: SecretKey, c: int | NDArray[np.object_], delta: float
) -> ArrayLike:
    return decode(params, decrypt(params, sk, c), delta)


def _encrypt(params: PublicParameters, pk: PublicKey, m: int) -> int:
    r = ru.get_rand(0, params.n)
    while gcd(r, params.n) != 1:
        r = ru.get_rand(0, params.n)

    return (
        pow(pk.g, m, params.n_square) * pow(r, params.n, params.n_square)
    ) % params.n_square


def _decrypt(params: PublicParameters, sk: SecretKey, c: int) -> int:
    return (((pow(c, sk.lmd, params.n_square) - 1) // params.n) * sk.mu) % params.n


def _add(params: PublicParameters, c1: int, c2: int) -> int:
    return (c1 * c2) % params.n_square


def _int_mult(params: PublicParameters, m: int, c: int) -> int:
    return pow(c, m, params.n_square)


def _encode(params: PublicParameters, x: float, delta: float) -> int:
    m = floor(x / delta + 0.5)

    if m < -((params.n - 1) // 2):
        raise exceptions.EncodingError("Underflow")

    elif m > (params.n // 2):
        raise exceptions.EncodingError("Overflow")

    else:
        return m % params.n


def _decode(params: PublicParameters, m: int, delta: float) -> float:
    return mu.min_residue(m, params.n) * delta
