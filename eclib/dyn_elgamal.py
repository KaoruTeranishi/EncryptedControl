#! /usr/bin/env python3

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

import eclib.randutils as ru
from eclib import elgamal, exceptions
from eclib.elgamal import PublicKey, PublicParameters, SecretKey


@dataclass(slots=True)
class Token:
    s: int
    h: int


def keygen(bit_length: int) -> tuple[PublicParameters, PublicKey, SecretKey]:
    return elgamal.keygen(bit_length)


def encrypt(params: PublicParameters, pk: PublicKey, m: int) -> NDArray[np.object_]:
    return elgamal.encrypt(params, pk, m)


def decrypt(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_]
) -> ArrayLike:
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
    params: PublicParameters, pk: PublicKey, x: ArrayLike, delta: float
) -> NDArray[np.object_]:
    return elgamal.enc(params, pk, x, delta)


def dec(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_], delta: float
) -> ArrayLike:
    return elgamal.dec(params, sk, c, delta)


def dec_add(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_], delta: float
) -> ArrayLike:
    return elgamal.dec_add(params, sk, c, delta)


def update_key(
    params: PublicParameters, pk: PublicKey, sk: SecretKey
) -> tuple[PublicKey, SecretKey, Token]:
    t = Token(ru.get_rand(1, params.q), pk.h)
    pk.h = pk.h * pow(params.g, t.s, params.p) % params.p
    sk.s = (sk.s + t.s) % params.q

    return pk, sk, t


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
    c[0] = (c[0] * pow(params.g, r, params.p)) % params.p

    return np.array(
        [
            c[0],
            (pow(c[0], t.s, params.p) * c[1] * pow(t.h, r, params.p)) % params.p,
        ],
        dtype=object,
    )
