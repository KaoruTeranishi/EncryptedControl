#! /usr/bin/env python3

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eclib import dyn_elgamal, elgamal, gsw, gsw_lwe, paillier, regev


class Plant:
    def __init__(
        self,
        A: ArrayLike,
        B: ArrayLike,
        C: ArrayLike,
        D: ArrayLike,
        x0: Optional[ArrayLike] = None,
    ):
        A_ = np.asarray(A, dtype=np.float64)

        match A_.ndim:
            case 0:
                self.A = A_.reshape(1, 1)

            case 2 if A_.shape[0] == A_.shape[1]:
                self.A = A_

            case _:
                raise ValueError

        B_ = np.asarray(B, dtype=np.float64)

        match B_.ndim:
            case 0 if self.A.shape[0] == 1:
                self.B = B_.reshape(1, 1)

            case 1 if self.A.shape[1] == 1:
                self.B = B_.reshape(1, len(B_))

            case 2 if self.A.shape[1] == B_.shape[0]:
                self.B = B_

            case _:
                raise ValueError

        C_ = np.asarray(C, dtype=np.float64)

        match C_.ndim:
            case 0 if self.A.shape[0] == 1:
                self.C = C_.reshape(1, 1)

            case 1 if self.A.shape[1] == len(C_):
                self.C = C_.reshape(1, len(C_))

            case 2 if self.A.shape[1] == C_.shape[1]:
                self.C = C_

            case _:
                raise ValueError

        D_ = np.asarray(D, dtype=np.float64)

        match D_.ndim:
            case 0 if self.B.shape[1] == 1 and self.C.shape[0] == 1:
                self.D = D_.reshape(1, 1)

            case 1 if self.B.shape[1] == len(D_) and self.C.shape[0] == 1:
                self.D = D_.reshape(1, len(D_))

            case 2 if (
                self.B.shape[1] == D_.shape[1] and self.C.shape[0] == D_.shape[0]
            ):
                self.D = D_

            case _:
                raise ValueError

        if x0 is None:
            self.state = np.zeros(self.A.shape[0], dtype=np.float64)

        else:
            x0_ = np.asarray(x0, dtype=np.float64)

            match x0_.ndim:
                case 0 if self.A.shape[0] == 1:
                    self.state = x0_.reshape(1)

                case 1 if self.A.shape[1] == x0_.shape[0]:
                    self.state = x0_

                case 2 if (self.A.shape[1] == x0_.shape[0] and x0_.shape[1] == 1):
                    self.state = x0_.reshape(-1)

                case _:
                    raise ValueError

        self.input = np.zeros(self.B.shape[1], dtype=np.float64)
        self.output = np.zeros(self.C.shape[0], dtype=np.float64)

    def update(self) -> None:
        self.state = self.A @ self.state + self.B @ self.input

    def reset(
        self,
        state: Optional[ArrayLike] = None,
        input: Optional[ArrayLike] = None,
        output: Optional[ArrayLike] = None,
    ) -> None:
        if state is None:
            self.state = np.zeros(self.A.shape[0], dtype=np.float64)

        else:
            x = np.asarray(state, dtype=np.float64)

            match x.ndim:
                case 0 if self.A.shape[0] == 1:
                    self.state = x.reshape(1)

                case 1 if self.A.shape[1] == x.shape[0]:
                    self.state = x

                case 2 if (self.A.shape[1] == x.shape[0] and x.shape[1] == 1):
                    self.state = x.reshape(-1)

                case _:
                    raise ValueError

        if input is None:
            self.input = np.zeros(self.B.shape[1], dtype=np.float64)

        else:
            u = np.asarray(input, dtype=np.float64)

            match u.ndim:
                case 0 if self.B.shape[1] == 1:
                    self.input = u.reshape(1)

                case 1 if self.B.shape[1] == u.shape[0]:
                    self.input = u

                case 2 if (self.B.shape[1] == u.shape[0] and u.shape[1] == 1):
                    self.input = u.reshape(-1)

                case _:
                    raise ValueError

        if output is None:
            self.output = np.zeros(self.C.shape[1], dtype=np.float64)

        else:
            y = np.asarray(output, dtype=np.float64)

            match y.ndim:
                case 0 if self.C.shape[1] == 1:
                    self.output = y.reshape(1)

                case 1 if self.C.shape[1] == y.shape[0]:
                    self.output = y

                case 2 if (self.C.shape[1] == y.shape[0] and y.shape[1] == 1):
                    self.output = y.reshape(-1)

                case _:
                    raise ValueError


class Operator:
    def __init__(
        self,
        scheme: Optional[str] = None,
        params: Optional[
            elgamal.PublicParameters
            | dyn_elgamal.PublicParameters
            | paillier.PublicParameters
            | regev.PublicParameters
            | gsw.PublicParameters
            | gsw_lwe.PublicParameters
        ] = None,
        pk: Optional[
            elgamal.PublicKey
            | dyn_elgamal.PublicKey
            | paillier.PublicKey
            | regev.PublicKey
            | gsw.PublicKey
            | gsw_lwe.PublicKey
        ] = None,
        delta: Optional[float] = None,
    ):
        self.scheme = scheme
        self.params = params
        self.pk = pk
        self.delta = delta

    def get_enc_reference(self, reference: ArrayLike) -> int | NDArray[np.object_]:
        if not isinstance(self.delta, float):
            raise TypeError

        match self.scheme:
            case "elgamal" if (
                isinstance(self.params, elgamal.PublicParameters)
                and isinstance(self.pk, elgamal.PublicKey)
            ):
                return elgamal.enc(self.params, self.pk, reference, self.delta)

            case "dyn_elgamal" if (
                isinstance(self.params, dyn_elgamal.PublicParameters)
                and isinstance(self.pk, dyn_elgamal.PublicKey)
            ):
                return dyn_elgamal.enc(self.params, self.pk, reference, self.delta)

            case "paillier" if (
                isinstance(self.params, paillier.PublicParameters)
                and isinstance(self.pk, paillier.PublicKey)
            ):
                return paillier.enc(self.params, self.pk, reference, self.delta)

            case "regev" if (
                isinstance(self.params, regev.PublicParameters)
                and isinstance(self.pk, regev.PublicKey)
            ):
                return regev.enc(self.params, self.pk, reference, self.delta)

            case "gsw" if (
                isinstance(self.params, gsw.PublicParameters)
                and isinstance(self.pk, gsw.PublicKey)
            ):
                return gsw.enc(self.params, self.pk, reference, self.delta)

            case "gsw_lwe" if (
                isinstance(self.params, gsw_lwe.PublicParameters)
                and isinstance(self.pk, gsw_lwe.PublicKey)
            ):
                return gsw_lwe.enc(self.params, self.pk, reference, self.delta)

            case _:
                raise TypeError


class Sensor:
    def __init__(
        self,
        scheme: Optional[str] = None,
        params: Optional[
            elgamal.PublicParameters
            | dyn_elgamal.PublicParameters
            | paillier.PublicParameters
            | regev.PublicParameters
            | gsw.PublicParameters
            | gsw_lwe.PublicParameters
        ] = None,
        pk: Optional[
            elgamal.PublicKey
            | dyn_elgamal.PublicKey
            | paillier.PublicKey
            | regev.PublicKey
            | gsw.PublicKey
            | gsw_lwe.PublicKey
        ] = None,
        delta: Optional[float] = None,
    ):
        self.scheme = scheme
        self.params = params
        self.pk = pk
        self.delta = delta

    def get_output(self, plant: Plant) -> ArrayLike:
        plant.output = plant.C @ plant.state + plant.D @ plant.input

        if plant.output.shape[0] == 1:
            return plant.output.item()

        else:
            return plant.output

    def get_enc_output(self, plant: Plant) -> int | NDArray[np.object_]:
        plant.output = plant.C @ plant.state + plant.D @ plant.input

        if not isinstance(self.delta, float):
            raise TypeError

        match self.scheme:
            case "elgamal" if (
                isinstance(self.params, elgamal.PublicParameters)
                and isinstance(self.pk, elgamal.PublicKey)
            ):
                return elgamal.enc(self.params, self.pk, plant.output, self.delta)

            case "dyn_elgamal" if (
                isinstance(self.params, dyn_elgamal.PublicParameters)
                and isinstance(self.pk, dyn_elgamal.PublicKey)
            ):
                return dyn_elgamal.enc(self.params, self.pk, plant.output, self.delta)

            case "paillier" if (
                isinstance(self.params, paillier.PublicParameters)
                and isinstance(self.pk, paillier.PublicKey)
            ):
                return paillier.enc(self.params, self.pk, plant.output, self.delta)

            case "regev" if (
                isinstance(self.params, regev.PublicParameters)
                and isinstance(self.pk, regev.PublicKey)
            ):
                return regev.enc(self.params, self.pk, plant.output, self.delta)

            case "gsw" if (
                isinstance(self.params, gsw.PublicParameters)
                and isinstance(self.pk, gsw.PublicKey)
            ):
                return gsw.enc(self.params, self.pk, plant.output, self.delta)

            case "gsw_lwe" if (
                isinstance(self.params, gsw_lwe.PublicParameters)
                and isinstance(self.pk, gsw_lwe.PublicKey)
            ):
                return gsw_lwe.enc(self.params, self.pk, plant.output, self.delta)

            case _:
                raise TypeError


class Actuator:
    def __init__(
        self,
        scheme: Optional[str] = None,
        params: Optional[
            elgamal.PublicParameters
            | dyn_elgamal.PublicParameters
            | paillier.PublicParameters
            | regev.PublicParameters
            | gsw.PublicParameters
            | gsw_lwe.PublicParameters
        ] = None,
        pk: Optional[
            elgamal.PublicKey
            | dyn_elgamal.PublicKey
            | paillier.PublicKey
            | regev.PublicKey
            | gsw.PublicKey
            | gsw_lwe.PublicKey
        ] = None,
        sk: Optional[
            elgamal.SecretKey
            | dyn_elgamal.SecretKey
            | paillier.SecretKey
            | regev.SecretKey
            | gsw.SecretKey
            | gsw_lwe.SecretKey
        ] = None,
        delta_enc: Optional[float] = None,
        delta_dec: Optional[float] = None,
    ):
        self.scheme = scheme
        self.params = params
        self.pk = pk
        self.sk = sk
        self.delta_enc = delta_enc
        self.delta_dec = delta_dec

    def set_input(self, plant: Plant, input: ArrayLike) -> None:
        u = np.asarray(input, dtype=np.float64)

        match u.ndim:
            case 0 if plant.B.shape[1] == 1:
                plant.input = u.reshape(1)

            case 1 if plant.B.shape[1] == u.shape[0]:
                plant.input = u

            case 2 if plant.B.shape[1] == u.shape[0]:
                plant.input = u.reshape(-1)

            case _:
                raise ValueError

    def set_enc_input(self, plant: Plant, input: int | NDArray[np.object_]) -> None:
        if not isinstance(self.delta_dec, float):
            raise TypeError

        match self.scheme:
            case "elgamal" if (
                isinstance(self.params, elgamal.PublicParameters)
                and isinstance(self.sk, elgamal.SecretKey)
            ):
                self.set_input(
                    plant,
                    elgamal.dec_add(
                        self.params,
                        self.sk,
                        np.asarray(input, dtype=object),
                        self.delta_dec,
                    ),
                )

            case "dyn_elgamal" if (
                isinstance(self.params, dyn_elgamal.PublicParameters)
                and isinstance(self.sk, dyn_elgamal.SecretKey)
            ):
                self.set_input(
                    plant,
                    dyn_elgamal.dec_add(
                        self.params,
                        self.sk,
                        np.asarray(input, dtype=object),
                        self.delta_dec,
                    ),
                )

            case "paillier" if (
                isinstance(self.params, paillier.PublicParameters)
                and isinstance(self.sk, paillier.SecretKey)
            ):
                self.set_input(
                    plant, paillier.dec(self.params, self.sk, input, self.delta_dec)
                )

            case "regev" if (
                isinstance(self.params, regev.PublicParameters)
                and isinstance(self.sk, regev.SecretKey)
            ):
                self.set_input(
                    plant,
                    regev.dec(
                        self.params,
                        self.sk,
                        np.asarray(input, dtype=object),
                        self.delta_dec,
                    ),
                )

            case "gsw" if (
                isinstance(self.params, gsw.PublicParameters)
                and isinstance(self.sk, gsw.SecretKey)
            ):
                self.set_input(
                    plant,
                    gsw.dec(
                        self.params,
                        self.sk,
                        np.asarray(input, dtype=object),
                        self.delta_dec,
                    ),
                )

            case "gsw_lwe" if (
                isinstance(self.params, gsw_lwe.PublicParameters)
                and isinstance(self.sk, gsw_lwe.SecretKey)
            ):
                self.set_input(
                    plant,
                    gsw_lwe.dec(
                        self.params,
                        self.sk,
                        np.asarray(input, dtype=object),
                        self.delta_dec,
                    ),
                )

            case _:
                raise TypeError

    def re_enc_state(
        self, controller_state: NDArray[np.object_]
    ) -> NDArray[np.object_]:
        if not isinstance(self.delta_enc, float) or not isinstance(
            self.delta_dec, float
        ):
            raise TypeError

        match self.scheme:
            case "elgamal" if (
                isinstance(self.params, elgamal.PublicParameters)
                and isinstance(self.pk, elgamal.PublicKey)
                and isinstance(self.sk, elgamal.SecretKey)
            ):
                return elgamal.enc(
                    self.params,
                    self.pk,
                    elgamal.dec_add(
                        self.params, self.sk, controller_state, self.delta_dec
                    ),
                    self.delta_enc,
                )

            case "dyn_elgamal" if (
                isinstance(self.params, dyn_elgamal.PublicParameters)
                and isinstance(self.pk, dyn_elgamal.PublicKey)
                and isinstance(self.sk, dyn_elgamal.SecretKey)
            ):
                return dyn_elgamal.enc(
                    self.params,
                    self.pk,
                    dyn_elgamal.dec_add(
                        self.params, self.sk, controller_state, self.delta_dec
                    ),
                    self.delta_enc,
                )

            case "paillier" if (
                isinstance(self.params, paillier.PublicParameters)
                and isinstance(self.pk, paillier.PublicKey)
                and isinstance(self.sk, paillier.SecretKey)
            ):
                return np.asarray(
                    paillier.enc(
                        self.params,
                        self.pk,
                        paillier.dec(
                            self.params, self.sk, controller_state, self.delta_dec
                        ),
                        self.delta_enc,
                    ),
                    dtype=object,
                )

            case "regev" if (
                isinstance(self.params, regev.PublicParameters)
                and isinstance(self.pk, regev.PublicKey)
                and isinstance(self.sk, regev.SecretKey)
            ):
                return regev.enc(
                    self.params,
                    self.pk,
                    regev.dec(self.params, self.sk, controller_state, self.delta_dec),
                    self.delta_enc,
                )

            case "gsw" if (
                isinstance(self.params, gsw.PublicParameters)
                and isinstance(self.pk, gsw.PublicKey)
                and isinstance(self.sk, gsw.SecretKey)
            ):
                return gsw.enc(
                    self.params,
                    self.pk,
                    gsw.dec(self.params, self.sk, controller_state, self.delta_dec),
                    self.delta_enc,
                )

            case "gsw_lwe" if (
                isinstance(self.params, gsw_lwe.PublicParameters)
                and isinstance(self.pk, gsw_lwe.PublicKey)
                and isinstance(self.sk, gsw_lwe.SecretKey)
            ):
                return gsw_lwe.enc(
                    self.params,
                    self.pk,
                    gsw_lwe.dec(self.params, self.sk, controller_state, self.delta_dec),
                    self.delta_enc,
                )

            case _:
                raise TypeError


class Controller:
    def __init__(
        self,
        A: ArrayLike,
        B: ArrayLike,
        C: ArrayLike,
        D: ArrayLike,
        E: Optional[ArrayLike] = None,
        F: Optional[ArrayLike] = None,
        x0: Optional[ArrayLike] = None,
    ):
        A_ = np.asarray(A, dtype=np.float64)

        match A_.ndim:
            case 0:
                self.A = A_.reshape(1, 1)

            case 2 if A_.shape[0] == A_.shape[1]:
                self.A = A_

            case _:
                raise ValueError

        B_ = np.asarray(B, dtype=np.float64)

        match B_.ndim:
            case 0 if self.A.shape[0] == 1:
                self.B = B_.reshape(1, 1)

            case 1 if self.A.shape[1] == 1:
                self.B = B_.reshape(1, len(B_))

            case 2 if self.A.shape[1] == B_.shape[0]:
                self.B = B_

            case _:
                raise ValueError

        C_ = np.asarray(C, dtype=np.float64)

        match C_.ndim:
            case 0 if self.A.shape[0] == 1:
                self.C = C_.reshape(1, 1)

            case 1 if self.A.shape[1] == len(C_):
                self.C = C_.reshape(1, len(C_))

            case 2 if self.A.shape[1] == C_.shape[1]:
                self.C = C_

            case _:
                raise ValueError

        D_ = np.asarray(D, dtype=np.float64)

        match D_.ndim:
            case 0 if self.B.shape[1] == 1 and self.C.shape[0] == 1:
                self.D = D_.reshape(1, 1)

            case 1 if self.B.shape[1] == len(D_) and self.C.shape[0] == 1:
                self.D = D_.reshape(1, len(D_))

            case 2 if (
                self.B.shape[1] == D_.shape[1] and self.C.shape[0] == D_.shape[0]
            ):
                self.D = D_

            case _:
                raise ValueError

        if E is None:
            self.E = np.zeros([self.A.shape[0], 1], dtype=np.float64)

        else:
            E_ = np.asarray(E, dtype=np.float64)

            match E_.ndim:
                case 0 if self.A.shape[0] == 1:
                    self.E = E_.reshape(1, 1)

                case 1 if self.A.shape[1] == 1:
                    self.E = E_.reshape(1, len(E_))

                case 2 if self.A.shape[1] == E_.shape[0]:
                    self.E = E_

                case _:
                    raise ValueError

        if F is None:
            self.F = np.zeros([self.C.shape[0], 1], dtype=np.float64)

        else:
            F_ = np.asarray(F, dtype=np.float64)

            match F_.ndim:
                case 0 if self.E.shape[1] == 1 and self.C.shape[0] == 1:
                    self.F = F_.reshape(1, 1)

                case 1 if self.E.shape[1] == len(F_) and self.C.shape[0] == 1:
                    self.F = D_.reshape(1, len(F_))

                case 2 if (
                    self.E.shape[1] == F_.shape[1] and self.C.shape[0] == F_.shape[0]
                ):
                    self.F = F_

                case _:
                    raise ValueError

        if x0 is None:
            self.state = np.zeros(self.A.shape[0], dtype=np.float64)

        else:
            x0_ = np.asarray(x0, dtype=np.float64)

            match x0_.ndim:
                case 0 if self.A.shape[0] == 1:
                    self.state = x0_.reshape(1)

                case 1 if self.A.shape[1] == x0_.shape[0]:
                    self.state = x0_

                case 2 if (self.A.shape[1] == x0_.shape[0] and x0_.shape[1] == 1):
                    self.state = x0_.reshape(-1)

                case _:
                    raise ValueError

        self.input = np.zeros(self.B.shape[1], dtype=np.float64)
        self.output = np.zeros(self.C.shape[0], dtype=np.float64)
        self.reference = np.zeros(self.E.shape[1], dtype=np.float64)

    def reset(
        self,
        state: Optional[ArrayLike] = None,
        input: Optional[ArrayLike] = None,
        output: Optional[ArrayLike] = None,
        reference: Optional[ArrayLike] = None,
    ) -> None:
        if state is None:
            self.state = np.zeros(self.A.shape[0], dtype=np.float64)

        else:
            x = np.asarray(state, dtype=np.float64)

            match x.ndim:
                case 0 if self.A.shape[0] == 1:
                    self.state = x.reshape(1)

                case 1 if self.A.shape[1] == x.shape[0]:
                    self.state = x

                case 2 if (self.A.shape[1] == x.shape[0] and x.shape[1] == 1):
                    self.state = x.reshape(-1)

                case _:
                    raise ValueError

        if input is None:
            self.input = np.zeros(self.B.shape[1], dtype=np.float64)

        else:
            y = np.asarray(input, dtype=np.float64)

            match y.ndim:
                case 0 if self.B.shape[1] == 1:
                    self.input = y.reshape(1)

                case 1 if self.B.shape[1] == y.shape[0]:
                    self.input = y

                case 2 if (self.B.shape[1] == y.shape[0] and y.shape[1] == 1):
                    self.input = y.reshape(-1)

                case _:
                    raise ValueError

        if output is None:
            self.output = np.zeros(self.C.shape[0], dtype=np.float64)

        else:
            u = np.asarray(output, dtype=np.float64)

            match u.ndim:
                case 0 if self.C.shape[0] == 1:
                    self.output = u.reshape(1)

                case 1 if self.C.shape[0] == u.shape[0]:
                    self.output = u

                case 2 if (self.C.shape[0] == u.shape[0] and u.shape[1] == 1):
                    self.output = u.reshape(-1)

                case _:
                    raise ValueError

        if reference is None:
            self.reference = np.zeros(self.E.shape[1], dtype=np.float64)

        else:
            r = np.asarray(reference, dtype=np.float64)

            match r.ndim:
                case 0 if self.E.shape[1] == 1:
                    self.reference = r.reshape(1)

                case 1 if self.E.shape[1] == r.shape[0]:
                    self.reference = r

                case 2 if (self.E.shape[1] == r.shape[0] and r.shape[1] == 1):
                    self.reference = r.reshape(-1)

                case _:
                    raise ValueError

    def get_output(
        self, measurement: ArrayLike, reference: Optional[ArrayLike] = None
    ) -> ArrayLike:
        y = np.asarray(measurement, dtype=np.float64)

        match y.ndim:
            case 0 if self.B.shape[1] == 1:
                self.input = y.reshape(1)

            case 1 if self.B.shape[1] == y.shape[0]:
                self.input = y

            case 2 if self.B.shape[1] == y.shape[0]:
                self.input = y.reshape(-1)

            case _:
                raise ValueError

        if reference is not None:
            r = np.asarray(reference, dtype=np.float64)

            match r.ndim:
                case 0 if self.E.shape[1] == 1:
                    self.reference = r.reshape(1)

                case 1 if self.E.shape[1] == r.shape[0]:
                    self.reference = r

                case 2 if self.E.shape[1] == r.shape[0]:
                    self.reference = r.reshape(-1)

                case _:
                    raise ValueError

            self.output = (
                self.C @ self.state + self.D @ self.input + self.F @ self.reference
            )
            self.state = (
                self.A @ self.state + self.B @ self.input + self.E @ self.reference
            )

        else:
            self.output = self.C @ self.state + self.D @ self.input
            self.state = self.A @ self.state + self.B @ self.input

        if self.output.shape[0] == 1:
            return self.output.item()

        else:
            return self.output


class EncryptedController:
    def __init__(
        self,
        scheme: str,
        params: (
            elgamal.PublicParameters
            | dyn_elgamal.PublicParameters
            | paillier.PublicParameters
            | regev.PublicParameters
            | gsw.PublicParameters
            | gsw_lwe.PublicParameters
        ),
        pk: (
            elgamal.PublicKey
            | dyn_elgamal.PublicKey
            | paillier.PublicKey
            | regev.PublicKey
            | gsw.PublicKey
            | gsw_lwe.PublicKey
        ),
        controller: Controller,
        delta: float,
    ):

        self.scheme = scheme
        self.params = params

        match scheme:
            case "elgamal" if (
                isinstance(params, elgamal.PublicParameters)
                and isinstance(pk, elgamal.PublicKey)
            ):
                self.A = elgamal.enc(params, pk, controller.A, delta)
                self.B = elgamal.enc(params, pk, controller.B, delta)
                self.C = elgamal.enc(params, pk, controller.C, delta)
                self.D = elgamal.enc(params, pk, controller.D, delta)
                self.E = elgamal.enc(params, pk, controller.E, delta)
                self.F = elgamal.enc(params, pk, controller.F, delta)
                self.state = elgamal.enc(params, pk, controller.state, delta)
                self.input = elgamal.enc(params, pk, controller.input, delta)
                self.output = elgamal.enc(params, pk, controller.output, delta)
                self.reference = elgamal.enc(params, pk, controller.reference, delta)

            case "dyn_elgamal" if (
                isinstance(params, dyn_elgamal.PublicParameters)
                and isinstance(pk, dyn_elgamal.PublicKey)
            ):
                self.A = dyn_elgamal.enc(params, pk, controller.A, delta)
                self.B = dyn_elgamal.enc(params, pk, controller.B, delta)
                self.C = dyn_elgamal.enc(params, pk, controller.C, delta)
                self.D = dyn_elgamal.enc(params, pk, controller.D, delta)
                self.E = dyn_elgamal.enc(params, pk, controller.E, delta)
                self.F = dyn_elgamal.enc(params, pk, controller.F, delta)
                self.state = dyn_elgamal.enc(params, pk, controller.state, delta)
                self.input = dyn_elgamal.enc(params, pk, controller.input, delta)
                self.output = dyn_elgamal.enc(params, pk, controller.output, delta)
                self.reference = dyn_elgamal.enc(
                    params, pk, controller.reference, delta
                )

            case "paillier" if (
                isinstance(params, paillier.PublicParameters)
                and isinstance(pk, paillier.PublicKey)
            ):
                self.A = np.asarray(
                    paillier.encode(params, controller.A, delta), dtype=object
                )
                self.B = np.asarray(
                    paillier.encode(params, controller.B, delta), dtype=object
                )
                self.C = np.asarray(
                    paillier.encode(params, controller.C, delta), dtype=object
                )
                self.D = np.asarray(
                    paillier.encode(params, controller.D, delta), dtype=object
                )
                self.E = np.asarray(
                    paillier.encode(params, controller.E, delta), dtype=object
                )
                self.F = np.asarray(
                    paillier.encode(params, controller.F, delta), dtype=object
                )
                self.state = np.asarray(
                    paillier.enc(params, pk, controller.state, delta), dtype=object
                )
                self.input = np.asarray(
                    paillier.enc(params, pk, controller.input, delta), dtype=object
                )
                self.output = np.asarray(
                    paillier.enc(params, pk, controller.output, delta), dtype=object
                )
                self.reference = np.asarray(
                    paillier.enc(params, pk, controller.reference, delta), dtype=object
                )

            case "regev" if (
                isinstance(params, regev.PublicParameters)
                and isinstance(pk, regev.PublicKey)
            ):
                self.A = np.asarray(
                    regev.encode(params, controller.A, delta), dtype=object
                )
                self.B = np.asarray(
                    regev.encode(params, controller.B, delta), dtype=object
                )
                self.C = np.asarray(
                    regev.encode(params, controller.C, delta), dtype=object
                )
                self.D = np.asarray(
                    regev.encode(params, controller.D, delta), dtype=object
                )
                self.E = np.asarray(
                    regev.encode(params, controller.E, delta), dtype=object
                )
                self.F = np.asarray(
                    regev.encode(params, controller.F, delta), dtype=object
                )
                self.state = regev.enc(params, pk, controller.state, delta)
                self.input = regev.enc(params, pk, controller.input, delta)
                self.output = regev.enc(params, pk, controller.output, delta)
                self.reference = regev.enc(params, pk, controller.reference, delta)

            case "gsw" if (
                isinstance(params, gsw.PublicParameters)
                and isinstance(pk, gsw.PublicKey)
            ):
                self.A = gsw.enc(params, pk, controller.A, delta)
                self.B = gsw.enc(params, pk, controller.B, delta)
                self.C = gsw.enc(params, pk, controller.C, delta)
                self.D = gsw.enc(params, pk, controller.D, delta)
                self.E = gsw.enc(params, pk, controller.E, delta)
                self.F = gsw.enc(params, pk, controller.F, delta)
                self.state = gsw.enc(params, pk, controller.state, delta)
                self.input = gsw.enc(params, pk, controller.input, delta)
                self.output = gsw.enc(params, pk, controller.output, delta)
                self.reference = gsw.enc(params, pk, controller.reference, delta)

            case "gsw_lwe" if (
                isinstance(params, gsw_lwe.PublicParameters)
                and isinstance(pk, gsw_lwe.PublicKey)
            ):
                self.A = gsw_lwe.enc_gsw(params, pk, controller.A, delta)
                self.B = gsw_lwe.enc_gsw(params, pk, controller.B, delta)
                self.C = gsw_lwe.enc_gsw(params, pk, controller.C, delta)
                self.D = gsw_lwe.enc_gsw(params, pk, controller.D, delta)
                self.E = gsw_lwe.enc_gsw(params, pk, controller.E, delta)
                self.F = gsw_lwe.enc_gsw(params, pk, controller.F, delta)
                self.state = gsw_lwe.enc(params, pk, controller.state, delta)
                self.input = gsw_lwe.enc(params, pk, controller.input, delta)
                self.output = gsw_lwe.enc(params, pk, controller.output, delta)
                self.reference = gsw_lwe.enc(params, pk, controller.reference, delta)

            case _:
                raise TypeError

    def get_enc_output(
        self,
        measurement: NDArray[np.object_],
        reference: Optional[NDArray[np.object_]] = None,
        controller_state: Optional[NDArray[np.object_]] = None,
    ) -> tuple[NDArray[np.object_], NDArray[np.object_]]:
        if self.input.shape == measurement.shape:
            self.input = measurement

        else:
            raise ValueError

        if reference is not None:
            if self.reference.shape == reference.shape:
                self.reference = reference

            else:
                raise ValueError

        if controller_state is not None:
            if self.state.shape == controller_state.shape:
                self.state = controller_state

            else:
                raise ValueError

        if reference is None:
            controller_params = np.concatenate(
                [
                    np.concatenate([self.A, self.B], axis=1),
                    np.concatenate([self.C, self.D], axis=1),
                ],
                axis=0,
            )
            controller_inputs = np.concatenate([self.state, self.input], axis=0)

        else:
            controller_params = np.concatenate(
                [
                    np.concatenate([self.A, self.B, self.E], axis=1),
                    np.concatenate([self.C, self.D, self.F], axis=1),
                ],
                axis=0,
            )
            controller_inputs = np.concatenate(
                [self.state, self.input, self.reference], axis=0
            )

        match self.scheme:
            case "elgamal" if isinstance(self.params, elgamal.PublicParameters):
                controller_outputs = elgamal.mult(
                    self.params, controller_params, controller_inputs
                )

            case "dyn_elgamal" if isinstance(self.params, dyn_elgamal.PublicParameters):
                controller_outputs = dyn_elgamal.mult(
                    self.params, controller_params, controller_inputs
                )

            case "paillier" if isinstance(self.params, paillier.PublicParameters):
                controller_outputs = np.asarray(
                    paillier.int_mult(
                        self.params, controller_params, controller_inputs
                    ),
                    dtype=object,
                )

            case "regev" if isinstance(self.params, regev.PublicParameters):
                controller_outputs = regev.int_mult(
                    self.params, controller_params, controller_inputs
                )

            case "gsw" if isinstance(self.params, gsw.PublicParameters):
                controller_outputs = gsw.mult(
                    self.params, controller_params, controller_inputs
                )

            case "gsw_lwe" if isinstance(self.params, gsw_lwe.PublicParameters):
                controller_outputs = gsw_lwe.mult(
                    self.params, controller_params, controller_inputs
                )

            case _:
                TypeError

        controller_state_update = controller_outputs[: self.A.shape[0]]
        self.output = controller_outputs[self.A.shape[0] :]

        return controller_state_update, self.output