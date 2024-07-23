#! /usr/bin/env python3

from dataclasses import dataclass
from math import floor

import numpy as np
from numpy.typing import ArrayLike, NDArray

import eclib.numutils as nu
import eclib.primeutils as pu
import eclib.randutils as ru
from eclib import exceptions


@dataclass(frozen=True, slots=True)
class PublicParameters:
    p: int
    q: int
    g: int


def keygen(bit_length: int) -> tuple[PublicParameters, int, int]:
    q, p = pu.get_safe_prime(bit_length)
    g = nu.get_generator(q, p)
    params = PublicParameters(p, q, g)

    sk = ru.get_rand(1, params.q)

    pk = pow(params.g, sk, params.p)

    return params, pk, sk


def encrypt(params: PublicParameters, pk: int, m: ArrayLike) -> NDArray[np.object_]:
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


def decrypt(params: PublicParameters, sk: int, c: NDArray[np.object_]) -> ArrayLike:
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
            raise exceptions.EncryptionError


def mult(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)

    match c1.ndim - 1:
        case 0:
            match c2.ndim - 1:
                case 0:
                    return _mult(params, c1, c2)

                case 1:
                    return np.array(
                        [_mult(params, c1, c2[i]) for i in range(c2.shape[0])],
                        dtype=object,
                    )

                case 2:
                    return np.array(
                        [
                            [_mult(params, c1, c2[i][j]) for j in range(c2.shape[1])]
                            for i in range(c2.shape[0])
                        ],
                        dtype=object,
                    )

                case _:
                    raise exceptions.HomomorphicOperationError

        case 1:
            if c1.shape == c2.shape:
                return np.array(
                    [_mult(params, c1[i], c2[i]) for i in range(c1.shape[0])],
                    dtype=object,
                )

            else:
                raise exceptions.HomomorphicOperationError

        case 2:
            match c2.ndim - 1:
                case 1:
                    if c1.shape[1] == c2.shape[0]:
                        return np.array(
                            [
                                [
                                    _mult(params, c1[i][j], c2[j])
                                    for j in range(c1.shape[1])
                                ]
                                for i in range(c1.shape[0])
                            ],
                            dtype=object,
                        )

                    else:
                        raise exceptions.HomomorphicOperationError

                case 2:
                    if c1.shape == c2.shape:
                        return np.array(
                            [
                                [
                                    _mult(params, c1[i][j], c2[i][j])
                                    for j in range(c1.shape[1])
                                ]
                                for i in range(c1.shape[0])
                            ],
                            dtype=object,
                        )

                    else:
                        raise exceptions.HomomorphicOperationError

                case _:
                    raise exceptions.HomomorphicOperationError

        case _:
            raise exceptions.HomomorphicOperationError


def encode(params: PublicParameters, x: ArrayLike, delta: float) -> ArrayLike:
    f = np.frompyfunc(_encode, 3, 1)
    return f(params, x, delta)


def decode(params: PublicParameters, m: ArrayLike, delta: float) -> ArrayLike:
    f = np.frompyfunc(_decode, 3, 1)
    return f(params, m, delta)


def enc(
    params: PublicParameters, pk: int, x: ArrayLike, delta: float
) -> NDArray[np.object_]:
    return encrypt(params, pk, encode(params, x, delta))


def dec(
    params: PublicParameters, sk: int, c: NDArray[np.object_], delta: float
) -> ArrayLike:
    return decode(params, decrypt(params, sk, c), delta)


def dec_add(
    params: PublicParameters, sk: int, c: NDArray[np.object_], delta: float
) -> ArrayLike:
    c = np.asarray(c)

    match c.ndim - 1:
        case 0:
            return dec(params, sk, c, delta)

        case 1:
            x = dec(params, sk, c, delta)
            return np.sum(x, axis=0)

        case 2:
            x = dec(params, sk, c, delta)
            return np.sum(x, axis=1)

        case _:
            raise exceptions.DecryptionError


def _encrypt(params: PublicParameters, pk: int, m: int) -> NDArray[np.object_]:
    r = ru.get_rand(1, params.q)

    return np.array(
        [pow(params.g, r, params.p), (m * pow(pk, r, params.p)) % params.p],
        dtype=object,
    )


def _decrypt(params: PublicParameters, sk: int, c: NDArray[np.object_]) -> int:
    return (pow(c[0], -sk, params.p) * c[1]) % params.p


def _mult(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    return np.array(
        [(c1[0] * c2[0]) % params.p, (c1[1] * c2[1]) % params.p], dtype=object
    )


def _encode(params: PublicParameters, x: float, delta: float) -> int:
    m = floor(x / delta + 0.5)
    first_decimal_place = (x / delta * 10) % 10

    if m < 0:
        if m < -params.q:
            raise exceptions.EncodingError("Underflow")
        else:
            m += params.p
    elif m > params.q:
        raise exceptions.EncodingError("Overflow")

    if x / delta == int(x / delta) or first_decimal_place >= 5:
        for i in range(params.q):
            if m - i > 0 and nu.is_element(m - i, params.q, params.p):
                return m - i
            elif m + i < params.p and nu.is_element(m + i, params.q, params.p):
                return m + i
    else:
        for i in range(params.q):
            if m + i < params.p and nu.is_element(m + i, params.q, params.p):
                return m + i
            elif m - i > 0 and nu.is_element(m - i, params.q, params.p):
                return m - i

    raise exceptions.EncodingError


def _decode(params: PublicParameters, m: int, delta: float) -> float:
    if m > params.q:
        return (m - params.p) * delta

    else:
        return m * delta
