#! /usr/bin/env python3

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

import eclib.randutils as ru
from eclib import elgamal, exceptions
from eclib.elgamal import PublicParameters


@dataclass(slots=True)
class Token:
    s: int
    h: int


def keygen(bit_length: int) -> tuple[PublicParameters, int, int]:
    return elgamal.keygen(bit_length)


def encrypt(params: PublicParameters, pk: int, m: int) -> NDArray[np.object_]:
    return elgamal.encrypt(params, pk, m)


def decrypt(params: PublicParameters, sk: int, c: NDArray[np.object_]) -> ArrayLike:
    return elgamal.decrypt(params, sk, c)


def mult(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    return elgamal.mult(params, c1, c2)


def encode(params: PublicParameters, x: ArrayLike, delta: float) -> ArrayLike:
    return elgamal.encode(params, x, delta)


def decode(params: PublicParameters, m: ArrayLike, delta: float) -> ArrayLike:
    return elgamal.decode(params, m, delta)


def enc(
    params: PublicParameters, pk: int, x: ArrayLike, delta: float
) -> NDArray[np.object_]:
    return elgamal.enc(params, pk, x, delta)


def dec(
    params: PublicParameters, sk: int, c: NDArray[np.object_], delta: float
) -> ArrayLike:
    return elgamal.dec(params, sk, c, delta)


def dec_add(
    params: PublicParameters, sk: int, c: NDArray[np.object_], delta: float
) -> ArrayLike:
    return elgamal.dec_add(params, sk, c, delta)


def update_key(params: PublicParameters, pk: int, sk: int) -> tuple[int, int, Token]:
    t = Token(ru.get_rand(1, params.q), pk)
    return (pk * pow(params.g, t.s, params.p)) % params.p, (sk + t.s) % params.q, t


def update_ct(
    params: PublicParameters, c: NDArray[np.object_], t: Token
) -> NDArray[np.object_]:
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
            raise exceptions.CiphertextUpdateError


def _update_ct(
    params: PublicParameters, c: NDArray[np.object_], t: Token
) -> NDArray[np.object_]:
    c = np.asarray(c, dtype=object)
    r = ru.get_rand(1, params.q)
    tmp = (c[0] * pow(params.g, r, params.p)) % params.p

    return np.array(
        [
            tmp,
            (pow(tmp, t.s, params.p) * c[1] * pow(t.h, r, params.p)) % params.p,
        ],
        dtype=object,
    )
