import sys

import matplotlib.pyplot as plt
import numpy as np
from control.matlab import dlqr

from eclib import dyn_elgamal, elgamal, figure, gsw, gsw_lwe, paillier, regev, system

args = sys.argv

if len(args) < 2:
    raise UserWarning("Encryption scheme must be specified.")

elif len(args) > 2:
    raise UserWarning("Too many arguments.")

scheme = args[1]
params: (
    elgamal.PublicParameters
    | dyn_elgamal.PublicParameters
    | paillier.PublicParameters
    | regev.PublicParameters
    | gsw.PublicParameters
    | gsw_lwe.PublicParameters
)
pk: (
    elgamal.PublicKey
    | dyn_elgamal.PublicKey
    | paillier.PublicKey
    | regev.PublicKey
    | gsw.PublicKey
    | gsw_lwe.PublicKey
)
sk: (
    elgamal.SecretKey
    | dyn_elgamal.SecretKey
    | paillier.SecretKey
    | regev.SecretKey
    | gsw.SecretKey
    | gsw_lwe.SecretKey
)
key_length: int
sec_params: tuple

match scheme:
    case "elgamal":
        key_length = 64
        s = 0.001
        params, pk, sk = elgamal.keygen(key_length)

    case "dyn_elgamal":
        key_length = 64
        s = 0.001
        params, pk, sk = dyn_elgamal.keygen(key_length)

    case "paillier":
        key_length = 64
        s = 0.01
        params, pk, sk = paillier.keygen(key_length)

    case "regev":
        sec_params = (10, pow(2, 32), pow(2, 64), 3.2)
        s = 0.001
        params, pk, sk = regev.keygen(*sec_params)

    case "gsw":
        sec_params = (3, pow(2, 32), 3.2)
        s = 0.01
        params, pk, sk = gsw.keygen(*sec_params, 1)

    case "gsw_lwe":
        sec_params = (10, pow(2, 20), pow(2, 32), 3.2)
        s = 0.01
        params, pk, sk = gsw_lwe.keygen(*sec_params)

    case _:
        raise UserWarning(
            "Implemented encryption schemes: "
            + "elgamal, "
            + "dyn_elgamal, "
            + "paillier, "
            + "regev, "
            + "gsw, "
            + "gsw_lwe"
        )

# simulation steps
t = np.arange(0, 100)

# plant
A = [
    [0.3547, -0.1567],
    [0.1254, 0.9817],
]
B = [
    [0.0313],
    [0.0037],
]
C = [
    [1, 0],
    [0, 1],
]
D = [
    [0],
    [0],
]
x0 = [-1, 2]

plant = system.Plant(A, B, C, D, x0)

n = plant.A.shape[0]
m = plant.B.shape[1]

# sensor
sensor = system.Sensor(scheme, params, pk, s)

# actuator
actuator = system.Actuator(scheme, params, pk, sk, s, s**2)

# controller
K, _, _ = dlqr(plant.A, plant.B, np.eye(n), np.eye(m))

Ac = 0
Bc = [0, 0]
Cc = 0
Dc = -K

controller = system.Controller(Ac, Bc, Cc, Dc)

encrypted_controller = system.EncryptedController(scheme, params, pk, controller, s)

# state log data
x = np.zeros([len(t), n])
x_ = np.zeros([len(t), n])
x_enc = np.zeros(len(t), dtype=object)

# input log data
u = np.zeros([len(t), m])
u_ = np.zeros([len(t), m])
u_enc = np.zeros(len(t), dtype=object)


# simulation (unencrypted)
for k in range(len(t)):
    # measure sensor data
    x[k] = sensor.get_output(plant)

    # compute control input
    u[k] = controller.get_output(x[k])

    # input control action
    actuator.set_input(plant, u[k])

    # update plant state
    plant.update()


plant.reset(x0)


# simulation (encrypted)
for k in range(len(t)):
    # measure sensor data
    x_enc[k] = sensor.get_enc_output(plant)

    # logging
    x_[k] = plant.state

    # compute control input
    _, u_enc[k] = encrypted_controller.get_enc_output(x_enc[k])

    # input control action
    actuator.set_enc_input(plant, u_enc[k])

    # logging
    u_[k] = plant.input

    # update plant state
    plant.update()


# figure
figure.setup()
orange, blue = figure.Colors.orange, figure.Colors.blue

plt.figure()
plt.step(t, u_, linestyle="-", color=blue, linewidth=1.0, label="encrypted")
plt.step(t, u, linestyle="--", color=orange, linewidth=1.0, label="unencrypted")
plt.xlabel("Step")
plt.ylabel(r"$u$")
plt.xlim(0, max(t) + 1)
plt.legend(loc="lower right")

plt.figure()
plt.step(t, x_, linestyle="-", color=blue, linewidth=1.0, label=["encrypted", ""])
plt.step(t, x, linestyle="--", color=orange, linewidth=1.0, label=["unencrypted", ""])
plt.xlabel("Step")
plt.ylabel(r"$x$")
plt.xlim(0, max(t) + 1)
plt.legend(loc="upper right")

plt.show()
