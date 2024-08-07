#! /usr/bin/env python3

"""Control system components.

This module defines classes for a control system, including a plant, sensor, actuator,
operator, and controller. The classes represent the components of a control system and
provide methods for fundamental operations in control systems. The classes also include
methods for encrypted control systems using various encryption schemes.

Classes
-------
- Plant
- Sensor
- Actuator
- Controller
- Operator
- EncryptedController
"""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eclib import dyn_elgamal, elgamal, gsw, gsw_lwe, paillier, regev


class Plant:
    """
    Represents a plant in a control system.

    Attributes
    ----------
    A : numpy.ndarray
        State matrix.
    B : numpy.ndarray
        Input matrix.
    C : numpy.ndarray
        Output matrix.
    D : numpy.ndarray
        Feedforward matrix.
    state : numpy.ndarray
        Current state.
    input : numpy.ndarray
        Current input.
    output : numpy.ndarray
        Current output.

    Note
    ----
    The plant is modeled as a linear time-invariant system with state-space
    representation:

        `x(t + 1) = A x(t) + B u(t)`

        `y(t) = C x(t) + D u(t)`

    where `x` is the state, `u` is the input, and `y` is the output.
    """

    A: NDArray
    B: NDArray
    C: NDArray
    D: NDArray
    state: NDArray
    input: NDArray
    output: NDArray

    def __init__(
        self,
        A: ArrayLike,
        B: ArrayLike,
        C: ArrayLike,
        D: ArrayLike,
        x0: Optional[ArrayLike] = None,
    ):
        """
        Initialize a new Plant object.

        Parameters
        ----------
        A : array_like
            State matrix.
        B : array_like
            Input matrix.
        C : array_like
            Output matrix.
        D : array_like
            Feedforward matrix.
        x0 : array_like, optional
            Initial state.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the dimensions of the matrices or state are invalid.

        Note
        ----
        `A`, `B`, `C`, and `D` must be `n x n`, `n x m`, `l x n`, and `l x m` matrices,
        respectively, where `n` is a state dimension, `m` is an input dimension, and
        `l` is an output dimension.

        If the matrices are 1D arrays, they are reshaped to 2D arrays with a single
        row or column as necessary. If the state is a 2D array with a single column,
        it is reshaped to a 1D array.

        If the initial state is not provided, it is set to a zero vector of the same
        dimension as the state matrix `A`.
        """

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
        """
        Updates the state of the plant based on the current state and input.

        Returns
        -------
        None
        """

        self.state = self.A @ self.state + self.B @ self.input

    def reset(
        self,
        state: Optional[ArrayLike] = None,
        input: Optional[ArrayLike] = None,
        output: Optional[ArrayLike] = None,
    ) -> None:
        """
        Resets the state, input, and output of the plant.

        Parameters
        ----------
        state : array_like, optional
            New state.
        input : array_like, optional
            New input.
        output : array_like, optional
            New output.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the dimensions of the state, input, or output are invalid.

        Note
        ----
        The dimensions of the state, input, and output must match the dimensions of the
        state matrix `A`, input matrix `B`, and output matrix `C`, respectively.

        If the state, input, or output is not provided, it is set to a zero vector of
        the appropriate dimension.
        """

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


class Sensor:
    """
    Represents a sensor in a control system.

    Attributes
    ----------
    scheme : str or None
        Encryption scheme used by the sensor.
    params : public parameters or None
        Cryptosystem parameters of the encryption scheme.
    pk : public key or None
        Public key of the encryption scheme.
    scale : float or None
        Scaling factor.
    """

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
        scale: Optional[float] = None,
    ):
        """
        Initialize a new Sensor object.

        Parameters
        ----------
        scheme : str, optional
            Encryption scheme used by the sensor.
        params : public parameters or None, optional
            Cryptosystem parameters of the encryption scheme.
        pk : public key or None, optional
            Public key of the encryption scheme.
        scale : float, optional
            Scaling factor.

        Returns
        -------
        None

        Note
        ----
        The encryption scheme must be one of the following:

            - "elgamal"
            - "dyn_elgamal"
            - "paillier"
            - "regev"
            - "gsw"
            - "gsw_lwe"
        """

        self.scheme = scheme
        self.params = params
        self.pk = pk
        self.scale = scale

    def get_output(self, plant: Plant) -> ArrayLike:
        """
        Gets a measurement output of a `plant`.

        Parameters
        ----------
        plant : Plant
            Plant whose output is to be measured.

        Returns
        -------
        array_like
            Measurement output of the plant.
        """

        plant.output = plant.C @ plant.state + plant.D @ plant.input

        if plant.output.shape[0] == 1:
            return plant.output.item()

        else:
            return plant.output

    def get_enc_output(self, plant: Plant) -> int | NDArray[np.object_]:
        """
        Gets and encrypts a measurement output of a `plant`.

        Parameters
        ----------
        plant : Plant
            Plant whose output is to be measured.

        Returns
        -------
        int or numpy.ndarray
            Encrypted measurement output of the plant.

        Raises
        ------
        TypeError
            If the encryption scheme, cryptosystem parameters, public key, or scaling
            factor is invalid.
        """

        plant.output = plant.C @ plant.state + plant.D @ plant.input

        if not isinstance(self.scale, float):
            raise TypeError

        match self.scheme:
            case "elgamal" if (
                isinstance(self.params, elgamal.PublicParameters)
                and isinstance(self.pk, elgamal.PublicKey)
            ):
                return elgamal.enc(self.params, self.pk, plant.output, self.scale)

            case "dyn_elgamal" if (
                isinstance(self.params, dyn_elgamal.PublicParameters)
                and isinstance(self.pk, dyn_elgamal.PublicKey)
            ):
                return dyn_elgamal.enc(self.params, self.pk, plant.output, self.scale)

            case "paillier" if (
                isinstance(self.params, paillier.PublicParameters)
                and isinstance(self.pk, paillier.PublicKey)
            ):
                return paillier.enc(self.params, self.pk, plant.output, self.scale)

            case "regev" if (
                isinstance(self.params, regev.PublicParameters)
                and isinstance(self.pk, regev.PublicKey)
            ):
                return regev.enc(self.params, self.pk, plant.output, self.scale)

            case "gsw" if (
                isinstance(self.params, gsw.PublicParameters)
                and isinstance(self.pk, gsw.PublicKey)
            ):
                return gsw.enc(self.params, self.pk, plant.output, self.scale)

            case "gsw_lwe" if (
                isinstance(self.params, gsw_lwe.PublicParameters)
                and isinstance(self.pk, gsw_lwe.PublicKey)
            ):
                return gsw_lwe.enc(self.params, self.pk, plant.output, self.scale)

            case _:
                raise TypeError


class Actuator:
    """
    Represents an actuator in a control system.

    Attributes
    ----------
    scheme : str or None
        Encryption scheme used by the actuator.
    params : public parameters or None
        Cryptosystem parameters of the encryption scheme.
    pk : public key or None
        Public key of the encryption scheme.
    sk : secret key or None
        Secret key of the encryption scheme.
    scale_enc : float or None
        Scaling factor for encoding and encryption.
    scale_dec : float or None
        Scaling factor for decoding and decryption.
    """

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
        scale_enc: Optional[float] = None,
        scale_dec: Optional[float] = None,
    ):
        """
        Initialize a new Actuator object.

        Parameters
        ----------
        scheme : str, optional
            Encryption scheme used by the actuator.
        params : public parameters or None, optional
            Cryptosystem parameters of the encryption scheme.
        pk : public key or None, optional
            Public key of the encryption scheme.
        sk : secret key or None, optional
            Secret key of the encryption scheme.
        scale_enc : float, optional
            Scaling factor for encoding and encryption.
        scale_dec : float, optional
            Scaling factor for decoding and decryption.

        Returns
        -------
        None

        Note
        ----
        The encryption scheme must be one of the following:

            - "elgamal"
            - "dyn_elgamal"
            - "paillier"
            - "regev"
            - "gsw"
            - "gsw_lwe"
        """

        self.scheme = scheme
        self.params = params
        self.pk = pk
        self.sk = sk
        self.scale_enc = scale_enc
        self.scale_dec = scale_dec

    def set_input(self, plant: Plant, input: ArrayLike) -> None:
        """
        Set a control input to a `plant`.

        Parameters
        ----------
        plant : Plant
            Plant whose control input is to be set.
        input : array_like
            Control input.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the dimension of the input is invalid.
        """

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
        """
        Decrypts and sets an encrypted control input to a `plant`.

        Parameters
        ----------
        plant : Plant
            Plant whose control input is to be set.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the encryption scheme, cryptosystem parameters, public key, or scaling
            factor is invalid.
        """

        if not isinstance(self.scale_dec, float):
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
                        self.scale_dec,
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
                        self.scale_dec,
                    ),
                )

            case "paillier" if (
                isinstance(self.params, paillier.PublicParameters)
                and isinstance(self.sk, paillier.SecretKey)
            ):
                self.set_input(
                    plant, paillier.dec(self.params, self.sk, input, self.scale_dec)
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
                        self.scale_dec,
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
                        self.scale_dec,
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
                        self.scale_dec,
                    ),
                )

            case _:
                raise TypeError

    def re_enc_state(
        self, controller_state: NDArray[np.object_]
    ) -> NDArray[np.object_]:
        """
        Re-encrypts an encrypted controller state using the specified encryption scheme.

        Parameters
        ----------
        controller_state : numpy.ndarray
            Encrypted controller state to be re-encrypted.

        Returns
        -------
        numpy.ndarray
            Re-encrypted controller state.

        Raises
        ------
        TypeError
            If the encryption scheme, cryptosystem parameters, public key, or scaling
            factor is invalid.

        Note
        ----
        This method is used for removing the accumulation of scaling factors in the
        controller state.
        """

        if not isinstance(self.scale_enc, float) or not isinstance(
            self.scale_dec, float
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
                        self.params, self.sk, controller_state, self.scale_dec
                    ),
                    self.scale_enc,
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
                        self.params, self.sk, controller_state, self.scale_dec
                    ),
                    self.scale_enc,
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
                            self.params, self.sk, controller_state, self.scale_dec
                        ),
                        self.scale_enc,
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
                    regev.dec(self.params, self.sk, controller_state, self.scale_dec),
                    self.scale_enc,
                )

            case "gsw" if (
                isinstance(self.params, gsw.PublicParameters)
                and isinstance(self.pk, gsw.PublicKey)
                and isinstance(self.sk, gsw.SecretKey)
            ):
                return gsw.enc(
                    self.params,
                    self.pk,
                    gsw.dec(self.params, self.sk, controller_state, self.scale_dec),
                    self.scale_enc,
                )

            case "gsw_lwe" if (
                isinstance(self.params, gsw_lwe.PublicParameters)
                and isinstance(self.pk, gsw_lwe.PublicKey)
                and isinstance(self.sk, gsw_lwe.SecretKey)
            ):
                return gsw_lwe.enc(
                    self.params,
                    self.pk,
                    gsw_lwe.dec(self.params, self.sk, controller_state, self.scale_dec),
                    self.scale_enc,
                )

            case _:
                raise TypeError


class Controller:
    """
    Represents a controller in a control system.

    Attributes
    ----------
    A : numpy.ndarray
        State matrix.
    B : numpy.ndarray
        Input matrix.
    C : numpy.ndarray
        Output matrix.
    D : numpy.ndarray
        Feedforward matrix.
    E : numpy.ndarray
        Reference input matrix.
    F : numpy.ndarray
        Reference feedforward matrix.
    state : numpy.ndarray
        Current state.
    input : numpy.ndarray
        Current input.
    output : numpy.ndarray
        Current output.
    reference : numpy.ndarray
        Current reference.

    Note
    ----
    The controller is modeled as a linear time-invariant system with state-space
    representation:

        `x(t + 1) = A x(t) + B y(t) + E r(t)`

        `u(t) = C x(t) + D y(t) + F r(t)`

    where `x` is the controller state, `y` is the plant output, `u` is the plant input,
    and `r` is the reference. Note that the matrices `A`, `B`, `C`, and `D` are not the
    same as those of a plant in general.
    """

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
        """
        Initialize a new Controller object.

        Parameters
        ----------
        A : array_like
            State matrix.
        B : array_like
            Input matrix.
        C : array_like
            Output matrix.
        D : array_like
            Feedforward matrix.
        E : array_like, optional
            Reference input matrix.
        F : array_like, optional
            Reference feedforward matrix.
        x0 : array_like, optional
            Initial state.

        Raises
        ------
        ValueError
            If the dimensions of the matrices or state are invalid.

        Note
        ----
        `A`, `B`, `C`, `D`, `E`, and `F` must be `nc x nc`, `nc x l`, `m x nc`,
        `m x l`, `nc x q`, and `m x q` matrices, respectively, where `nc` is a
        controller state dimension, `l` is a plant output dimension, `m` is a plant
        input dimension, and `q` is a reference dimension.

        If the matrices are 1D arrays, they are reshaped to 2D arrays with a single
        row or column as necessary. If the state is a 2D array with a single column,
        it is reshaped to a 1D array.

        If the matrices `E` and `F` are not provided, they are set to zero matrices of
        the appropriate dimensions.

        If the initial state is not provided, it is set to a zero vector of the same
        dimension as the state matrix `A`.
        """

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
        """
        Resets the state, input, output, and reference of the controller.

        Parameters
        ----------
        state : array_like, optional
            New state.
        input : array_like, optional
            New input.
        output : array_like, optional
            New output.
        reference : array_like, optional
            New reference.

        Raises
        ------
        ValueError
            If the dimensions of the new state, input, output, or reference are invalid.

        Returns
        -------
        None

        Note
        ----
        If the new state, input, output, or reference is not provided, it is set to a
        zero vector of the appropriate dimension.
        """

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
        """
        Updates the state of the controller and computes a control input to a plant
        based on a measurement and a reference.

        Parameters
        ----------
        measurement : array_like
            Measurement output of the plant.
        reference : array_like, optional
            Reference input to the controller.

        Returns
        -------
        array_like
            Control input to the plant.

        Raises
        ------
        ValueError
            If the dimensions of the measurement or reference are invalid.
        """

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


class Operator:
    """
    Represents an operator in a control system who gives an encrypted reference to an
    encrypted controller.

    Attributes
    ----------
    scheme : str or None
        Encryption scheme used by the operator.
    params : public parameters or None
        Cryptosystem parameters of the encryption scheme.
    pk : public key or None
        Public key of the encryption scheme.
    scale : float or None
        Scaling factor.
    """

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
        scale: Optional[float] = None,
    ):
        """
        Initialize a new Operator object.

        Parameters
        ----------
        scheme : str, optional
            Encryption scheme used by the operator.
        params : public parameters or None, optional
            Cryptosystem parameters of the encryption scheme.
        pk : public key or None, optional
            Public key of the encryption scheme.
        scale : float, optional
            Scaling factor.

        Returns
        -------
        None

        Note
        ----
        The encryption scheme must be one of the following:

            - "elgamal"
            - "dyn_elgamal"
            - "paillier"
            - "regev"
            - "gsw"
            - "gsw_lwe"
        """

        self.scheme = scheme
        self.params = params
        self.pk = pk
        self.scale = scale

    def get_enc_reference(self, reference: ArrayLike) -> int | NDArray[np.object_]:
        """
        Encrypts a reference using the specified encryption scheme.

        Parameters
        ----------
        reference : array_like
            Reference to be encrypted.

        Returns
        -------
        int or numpy.ndarray
            Encrypted reference.

        Raises
        ------
        TypeError
            If the encryption scheme, cryptosystem parameters, public key, or scaling
            factor is invalid.
        """

        if not isinstance(self.scale, float):
            raise TypeError

        match self.scheme:
            case "elgamal" if (
                isinstance(self.params, elgamal.PublicParameters)
                and isinstance(self.pk, elgamal.PublicKey)
            ):
                return elgamal.enc(self.params, self.pk, reference, self.scale)

            case "dyn_elgamal" if (
                isinstance(self.params, dyn_elgamal.PublicParameters)
                and isinstance(self.pk, dyn_elgamal.PublicKey)
            ):
                return dyn_elgamal.enc(self.params, self.pk, reference, self.scale)

            case "paillier" if (
                isinstance(self.params, paillier.PublicParameters)
                and isinstance(self.pk, paillier.PublicKey)
            ):
                return paillier.enc(self.params, self.pk, reference, self.scale)

            case "regev" if (
                isinstance(self.params, regev.PublicParameters)
                and isinstance(self.pk, regev.PublicKey)
            ):
                return regev.enc(self.params, self.pk, reference, self.scale)

            case "gsw" if (
                isinstance(self.params, gsw.PublicParameters)
                and isinstance(self.pk, gsw.PublicKey)
            ):
                return gsw.enc(self.params, self.pk, reference, self.scale)

            case "gsw_lwe" if (
                isinstance(self.params, gsw_lwe.PublicParameters)
                and isinstance(self.pk, gsw_lwe.PublicKey)
            ):
                return gsw_lwe.enc(self.params, self.pk, reference, self.scale)

            case _:
                raise TypeError


class EncryptedController:
    """
    Represents an encrypted controller in a control system.

    Attributes
    ----------
    scheme : str
        Encryption scheme used by the controller.
    params : public parameters
        Cryptosystem parameters of the encryption scheme.
    A : numpy.ndarray
        Encrypted state matrix.
    B : numpy.ndarray
        Encrypted input matrix.
    C : numpy.ndarray
        Encrypted output matrix.
    D : numpy.ndarray
        Encrypted feedforward matrix.
    E : numpy.ndarray
        Encrypted reference input matrix.
    F : numpy.ndarray
        Encrypted reference feedforward matrix.
    state : numpy.ndarray
        Encrypted current state.
    input : numpy.ndarray
        Encrypted current input.
    output : numpy.ndarray
        Encrypted current output.
    reference : numpy.ndarray
        Encrypted current reference.
    """

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
        scale: float,
    ):
        """
        Initialize a new EncryptedController object.

        Parameters
        ----------
        scheme : str
            Encryption scheme used by the controller.
        params : public parameters
            Cryptosystem parameters of the encryption scheme.
        pk : public key
            Public key of the encryption scheme.
        controller : Controller
            Controller to be encrypted.
        scale : float
            Scaling factor.

        Raises
        ------
        TypeError
            If the encryption scheme, cryptosystem parameters, public key, or scaling
            factor is invalid.

        Note
        ----
        The encryption scheme must be one of the following:

            - "elgamal"
            - "dyn_elgamal"
            - "paillier"
            - "regev"
            - "gsw"
            - "gsw_lwe"

        If the encryption scheme is "paillier" or "regev", the matrices `A`, `B`, `C`,
        `D`, `E`, and `F` are stored as plaintexts. Otherwise, they are stored as
        ciphertexts.

        If the encryption scheme is "gsw_lwe", the matrices `A`, `B`, `C`, `D`, `E`,
        and `F` are stored as GSW ciphertexts, and `state`, `input`, `output`, and
        `reference` are stored as Regev (LWE) ciphertexts.
        """

        self.scheme = scheme
        self.params = params

        match scheme:
            case "elgamal" if (
                isinstance(params, elgamal.PublicParameters)
                and isinstance(pk, elgamal.PublicKey)
            ):
                self.A = elgamal.enc(params, pk, controller.A, scale)
                self.B = elgamal.enc(params, pk, controller.B, scale)
                self.C = elgamal.enc(params, pk, controller.C, scale)
                self.D = elgamal.enc(params, pk, controller.D, scale)
                self.E = elgamal.enc(params, pk, controller.E, scale)
                self.F = elgamal.enc(params, pk, controller.F, scale)
                self.state = elgamal.enc(params, pk, controller.state, scale)
                self.input = elgamal.enc(params, pk, controller.input, scale)
                self.output = elgamal.enc(params, pk, controller.output, scale)
                self.reference = elgamal.enc(params, pk, controller.reference, scale)

            case "dyn_elgamal" if (
                isinstance(params, dyn_elgamal.PublicParameters)
                and isinstance(pk, dyn_elgamal.PublicKey)
            ):
                self.A = dyn_elgamal.enc(params, pk, controller.A, scale)
                self.B = dyn_elgamal.enc(params, pk, controller.B, scale)
                self.C = dyn_elgamal.enc(params, pk, controller.C, scale)
                self.D = dyn_elgamal.enc(params, pk, controller.D, scale)
                self.E = dyn_elgamal.enc(params, pk, controller.E, scale)
                self.F = dyn_elgamal.enc(params, pk, controller.F, scale)
                self.state = dyn_elgamal.enc(params, pk, controller.state, scale)
                self.input = dyn_elgamal.enc(params, pk, controller.input, scale)
                self.output = dyn_elgamal.enc(params, pk, controller.output, scale)
                self.reference = dyn_elgamal.enc(
                    params, pk, controller.reference, scale
                )

            case "paillier" if (
                isinstance(params, paillier.PublicParameters)
                and isinstance(pk, paillier.PublicKey)
            ):
                self.A = np.asarray(
                    paillier.encode(params, controller.A, scale), dtype=object
                )
                self.B = np.asarray(
                    paillier.encode(params, controller.B, scale), dtype=object
                )
                self.C = np.asarray(
                    paillier.encode(params, controller.C, scale), dtype=object
                )
                self.D = np.asarray(
                    paillier.encode(params, controller.D, scale), dtype=object
                )
                self.E = np.asarray(
                    paillier.encode(params, controller.E, scale), dtype=object
                )
                self.F = np.asarray(
                    paillier.encode(params, controller.F, scale), dtype=object
                )
                self.state = np.asarray(
                    paillier.enc(params, pk, controller.state, scale), dtype=object
                )
                self.input = np.asarray(
                    paillier.enc(params, pk, controller.input, scale), dtype=object
                )
                self.output = np.asarray(
                    paillier.enc(params, pk, controller.output, scale), dtype=object
                )
                self.reference = np.asarray(
                    paillier.enc(params, pk, controller.reference, scale), dtype=object
                )

            case "regev" if (
                isinstance(params, regev.PublicParameters)
                and isinstance(pk, regev.PublicKey)
            ):
                self.A = np.asarray(
                    regev.encode(params, controller.A, scale), dtype=object
                )
                self.B = np.asarray(
                    regev.encode(params, controller.B, scale), dtype=object
                )
                self.C = np.asarray(
                    regev.encode(params, controller.C, scale), dtype=object
                )
                self.D = np.asarray(
                    regev.encode(params, controller.D, scale), dtype=object
                )
                self.E = np.asarray(
                    regev.encode(params, controller.E, scale), dtype=object
                )
                self.F = np.asarray(
                    regev.encode(params, controller.F, scale), dtype=object
                )
                self.state = regev.enc(params, pk, controller.state, scale)
                self.input = regev.enc(params, pk, controller.input, scale)
                self.output = regev.enc(params, pk, controller.output, scale)
                self.reference = regev.enc(params, pk, controller.reference, scale)

            case "gsw" if (
                isinstance(params, gsw.PublicParameters)
                and isinstance(pk, gsw.PublicKey)
            ):
                self.A = gsw.enc(params, pk, controller.A, scale)
                self.B = gsw.enc(params, pk, controller.B, scale)
                self.C = gsw.enc(params, pk, controller.C, scale)
                self.D = gsw.enc(params, pk, controller.D, scale)
                self.E = gsw.enc(params, pk, controller.E, scale)
                self.F = gsw.enc(params, pk, controller.F, scale)
                self.state = gsw.enc(params, pk, controller.state, scale)
                self.input = gsw.enc(params, pk, controller.input, scale)
                self.output = gsw.enc(params, pk, controller.output, scale)
                self.reference = gsw.enc(params, pk, controller.reference, scale)

            case "gsw_lwe" if (
                isinstance(params, gsw_lwe.PublicParameters)
                and isinstance(pk, gsw_lwe.PublicKey)
            ):
                self.A = gsw_lwe.enc_gsw(params, pk, controller.A, scale)
                self.B = gsw_lwe.enc_gsw(params, pk, controller.B, scale)
                self.C = gsw_lwe.enc_gsw(params, pk, controller.C, scale)
                self.D = gsw_lwe.enc_gsw(params, pk, controller.D, scale)
                self.E = gsw_lwe.enc_gsw(params, pk, controller.E, scale)
                self.F = gsw_lwe.enc_gsw(params, pk, controller.F, scale)
                self.state = gsw_lwe.enc(params, pk, controller.state, scale)
                self.input = gsw_lwe.enc(params, pk, controller.input, scale)
                self.output = gsw_lwe.enc(params, pk, controller.output, scale)
                self.reference = gsw_lwe.enc(params, pk, controller.reference, scale)

            case _:
                raise TypeError

    def get_enc_output(
        self,
        measurement: NDArray[np.object_],
        reference: Optional[NDArray[np.object_]] = None,
        controller_state: Optional[NDArray[np.object_]] = None,
    ) -> tuple[NDArray[np.object_], NDArray[np.object_]]:
        """
        Computes an encrypted state update and an encrypted control input to a plant
        based on an encrypted measurement and an encrypted reference.

        Parameters
        ----------
        measurement : numpy.ndarray
            Encrypted measurement output of the plant.
        reference : numpy.ndarray, optional
            Encrypted reference input to the controller.
        controller_state : numpy.ndarray, optional
            Encrypted current controller state.

        Returns
        -------
        controller_state_update : numpy.ndarray
            Encrypted state update of the controller.
        output : numpy.ndarray
            Encrypted control input to the plant.

        Raises
        ------
        ValueError
            If the dimensions of the measurement, reference, or controller state are
            invalid.
        TypeError
            If the encryption scheme is unsupported.

        Note
        ----
        If the reference is not provided, the controller is assumed to be of the form

            `xc(t + 1) = A xc(t) + B y(t)`

            `u(t) = C xc(t) + D y(t)`

        Otherwise, the controller is assumed to be of the form

            `xc(t + 1) = A xc(t) + B y(t) + E r(t)`

            `u(t) = C xc(t) + D y(t) + F r(t)`

        where `xc` is the controller state, `y` is the plant output, `u` is the plant
        input, and `r` is the reference.

        If the encryption scheme is "elgamal" or "dyn_elgamal", only the element-wise
        product of the controller parameters and inputs is computed.
        """

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
                raise TypeError("Unsupported encryption scheme.")

        controller_state_update = controller_outputs[: self.A.shape[0]]
        self.output = controller_outputs[self.A.shape[0] :]

        return controller_state_update, self.output
