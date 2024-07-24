#! /usr/bin/env python3

from dataclasses import dataclass
from math import ceil, floor, log2
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

import eclib.modutils as mu
import eclib.randutils as ru
from eclib import exceptions
from eclib.regev import PublicKey, SecretKey


@dataclass(slots=True)
class PublicParameters:
    n: int
    q: int
    sigma: int
    m: int
    l: int
    N: int

    def __init__(self, n: int, q: int, sigma: int, m: Optional[int] = None):
        self.n = n
        self.q = q
        self.sigma = sigma
        self.l = ceil(log2(q))
        self.N = (n + 1) * self.l

        if m is None:
            self.m = 2 * n * ceil(log2(q))

        else:
            self.m = m


def keygen(n, q, sigma, m=None):
    params = PublicParameters(n, q, sigma, m)

    sk = SecretKey(params)

    pk = PublicKey(params, sk)

    return params, pk, sk


def encrypt(
    params: PublicParameters, pk: PublicKey, m: ArrayLike
) -> NDArray[np.object_]:
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
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_]
) -> ArrayLike:
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
            raise exceptions.DecryptionError


def add(
    params: PublicParameters,
    c1: NDArray[np.object_],
    c2: NDArray[np.object_],
) -> NDArray[np.object_]:
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
                raise exceptions.HomomorphicOperationError

    else:
        raise exceptions.HomomorphicOperationError


def elementwise_add(
    params: PublicParameters,
    c1: NDArray[np.object_],
    c2: NDArray[np.object_],
) -> NDArray[np.object_]:
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
        raise exceptions.HomomorphicOperationError


def mult(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
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
            raise exceptions.HomomorphicOperationError


def elementwise_mult(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
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
            raise exceptions.HomomorphicOperationError


def int_mult(
    params: PublicParameters, m: ArrayLike, c: NDArray[np.object_]
) -> NDArray[np.object_]:
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
            raise exceptions.HomomorphicOperationError


def elementwise_int_mult(
    params: PublicParameters, m: ArrayLike, c: NDArray[np.object_]
) -> NDArray[np.object_]:
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
            raise exceptions.HomomorphicOperationError


def encode(params: PublicParameters, x: ArrayLike, delta: float) -> ArrayLike:
    f = np.frompyfunc(_encode, 3, 1)
    return f(params, x, delta)


def decode(params: PublicParameters, m: ArrayLike, delta: float) -> ArrayLike:
    f = np.frompyfunc(_decode, 3, 1)
    return f(params, m, delta)


def enc(
    params: PublicParameters, pk: PublicKey, x: ArrayLike, delta: float
) -> NDArray[np.object_]:
    return encrypt(params, pk, encode(params, x, delta))


def dec(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_], delta: float
) -> ArrayLike:
    return decode(params, decrypt(params, sk, c), delta)


def _encrypt(params: PublicParameters, pk: PublicKey, m: int) -> NDArray[np.object_]:
    R = np.array(
        [[ru.get_rand(0, 2) for _ in range(params.N)] for _ in range(params.m)],
        dtype=object,
    )
    G = _gadget(params)

    return (pk.B @ R + m * G) % params.q


def _decrypt(params: PublicParameters, sk: SecretKey, c: NDArray[np.object_]) -> int:
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
    return (c1 + c2) % params.q


def _mult(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    bit_decomposed = np.block(
        [_bitdecomp(params, c2[:, i]) for i in range(c2.shape[1])]
    )
    return (c1 @ bit_decomposed) % params.q


def _int_mult(
    params: PublicParameters, m: int, c: NDArray[np.object_]
) -> NDArray[np.object_]:
    return (m * c) % params.q


def _encode(params: PublicParameters, x: float, delta: float) -> int:
    m = floor(x / delta + 0.5)

    if m < -((params.q - 1) // 2):
        raise exceptions.EncodingError("Underflow")

    elif m > (params.q // 2):
        raise exceptions.EncodingError("Overflow")

    else:
        return m % params.q


def _decode(params: PublicParameters, m: int, delta: float) -> float:
    return mu.min_residue(m, params.q) * delta


def _gadget(params: PublicParameters) -> NDArray[np.object_]:
    g = 2 ** np.arange(params.l, dtype=object)
    return np.kron(np.identity(params.n + 1, dtype=object), g)


def _bitdecomp(params: PublicParameters, v: ArrayLike) -> NDArray[np.object_]:
    v = np.asarray(v, dtype=object)

    if v.ndim != 1:
        raise ValueError

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
