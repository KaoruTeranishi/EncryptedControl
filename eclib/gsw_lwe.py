#! /usr/bin/env python3

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eclib import gsw, regev
from eclib.regev import PublicKey, SecretKey


@dataclass(slots=True)
class PublicParameters:
    lwe_params: regev.PublicParameters
    gsw_params: gsw.PublicParameters

    def __init__(self, n: int, t: int, q: int, sigma: float, m: Optional[int] = None):
        self.lwe_params = regev.PublicParameters(n, t, q, sigma, m)
        self.gsw_params = gsw.PublicParameters(n, q, sigma, m)


def keygen(n: int, t: int, q: int, sigma: float, m: Optional[int] = None):
    params = PublicParameters(n, t, q, sigma, m)

    sk = SecretKey(params.lwe_params)

    pk = PublicKey(params.lwe_params, sk)

    return params, pk, sk


def encrypt(
    params: PublicParameters, pk: PublicKey, m: ArrayLike
) -> NDArray[np.object_]:
    return regev.encrypt(params.lwe_params, pk, m)


def encrypt_gsw(
    params: PublicParameters, pk: PublicKey, m: ArrayLike
) -> NDArray[np.object_]:
    return gsw.encrypt(params.gsw_params, pk, m)


def decrypt(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_]
) -> ArrayLike:
    return regev.decrypt(params.lwe_params, sk, c)


def add(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    return regev.add(params.lwe_params, c1, c2)


def elementwise_add(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
    return regev.elementwise_add(params.lwe_params, c1, c2)


def mult(
    params: PublicParameters, c1: NDArray[np.object_], c2: NDArray[np.object_]
) -> NDArray[np.object_]:
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
    return gsw.elementwise_mult(params.gsw_params, c1, c2)


def int_mult(
    params: PublicParameters, m: ArrayLike, c: NDArray[np.object_]
) -> NDArray[np.object_]:
    return regev.int_mult(params.lwe_params, m, c)


def elementwise_int_mult(
    params: PublicParameters, m: ArrayLike, c: NDArray[np.object_]
) -> NDArray[np.object_]:
    return regev.elementwise_int_mult(params.lwe_params, m, c)


def encode(params: PublicParameters, x: ArrayLike, delta: float) -> ArrayLike:
    return regev.encode(params.lwe_params, x, delta)


def decode(params: PublicParameters, m: ArrayLike, delta: float) -> ArrayLike:
    return regev.decode(params.lwe_params, m, delta)


def enc(
    params: PublicParameters, pk: PublicKey, x: ArrayLike, delta: float
) -> NDArray[np.object_]:
    return regev.enc(params.lwe_params, pk, x, delta)


def enc_gsw(
    params: PublicParameters, pk: PublicKey, x: ArrayLike, delta: float
) -> NDArray[np.object_]:
    return gsw.enc(params.gsw_params, pk, x, delta)


def dec(
    params: PublicParameters, sk: SecretKey, c: NDArray[np.object_], delta: float
) -> ArrayLike:
    return regev.dec(params.lwe_params, sk, c, delta)
