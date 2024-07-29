import sys

import matplotlib.pyplot as plt
import numpy as np

import eclib.figsetup
from eclib import dyn_elgamal, elgamal, gsw, gsw_lwe, paillier, regev, system
from eclib.colors import Colors

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
security_params: tuple

match scheme:
    case "elgamal":
        key_length = 64
        delta = 0.001
        params, pk, sk = elgamal.keygen(key_length)

    case "dyn_elgamal":
        key_length = 64
        delta = 0.001
        params, pk, sk = dyn_elgamal.keygen(key_length)

    case "paillier":
        key_length = 64
        delta = 0.01
        params, pk, sk = paillier.keygen(key_length)

    case "regev":
        security_params = (10, pow(2, 32), pow(2, 64), 3.2)
        delta = 0.001
        params, pk, sk = regev.keygen(*security_params)

    case "gsw":
        security_params = (3, pow(2, 32), 3.2)
        delta = 0.01
        params, pk, sk = gsw.keygen(*security_params, 1)

    case "gsw_lwe":
        security_params = (10, pow(2, 20), pow(2, 32), 3.2)
        delta = 0.01
        params, pk, sk = gsw_lwe.keygen(*security_params)

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
C = [0, 1]
D = 0
x0 = [-1, 2]

plant = system.Plant(A, B, C, D, x0)

n = plant.A.shape[0]
m = plant.B.shape[1]
l = plant.C.shape[0]

# sensor
sensor = system.Sensor(scheme, params, pk, delta)

# actuator
actuator = system.Actuator(scheme, params, pk, sk, delta, delta**2)

# controller
Kp = 10
Ki = 10
Ts = 0.1

Ac = 1
Bc = -Ts
Cc = Ki
Dc = -Kp
Ec = Ts
Fc = Kp
xc0 = 0

controller = system.Controller(Ac, Bc, Cc, Dc, Ec, Fc, xc0)

encrypted_controller = system.EncryptedController(scheme, params, pk, controller, delta)

# operator
operator = system.Operator(scheme, params, pk, delta)

# input log data
u = np.zeros([len(t), m])
u_ = np.zeros([len(t), m])
u_enc = np.zeros(len(t), dtype=object)

# output log data
y = np.zeros([len(t), l])
y_ = np.zeros([len(t), l])
y_enc = np.zeros(len(t), dtype=object)

# reference log data
r = np.zeros([len(t), l])
r_enc = np.zeros(len(t), dtype=object)


# simulation (unencrypted)
for k in range(len(t)):
    # reference
    r[k] = 1

    # measure sensor data
    y[k] = sensor.get_output(plant)

    # compute control input
    u[k] = controller.get_output(y[k], r[k])

    # input control action
    actuator.set_input(plant, u[k])

    # update plant state
    plant.update()


plant.reset(x0)
state_update = encrypted_controller.state


# simulation (encrypted)
for k in range(len(t)):
    # encrypt reference
    r_enc[k] = operator.get_enc_reference(r[k])

    # measure sensor data
    y_enc[k] = sensor.get_enc_output(plant)

    # logging
    y_[k] = plant.output

    # compute control input
    state_update, u_enc[k] = encrypted_controller.get_enc_output(
        y_enc[k], r_enc[k], state_update
    )

    # state re-encryption
    state_update = actuator.re_enc_state(state_update)

    # input control action
    actuator.set_enc_input(plant, u_enc[k])

    # logging
    u_[k] = plant.input

    # update plant state
    plant.update()


# figure
eclib.figsetup.figure_setup()
orange, blue = Colors.orange, Colors.blue

plt.figure()
plt.step(t, u_, linestyle="-", color=blue, linewidth=1.0, label="encrypted")
plt.step(t, u, linestyle="--", color=orange, linewidth=1.0, label="unencrypted")
plt.xlabel("Step")
plt.ylabel(r"$u$")
plt.xlim(0, max(t) + 1)
plt.legend(loc="lower right")

plt.figure()
plt.step(t, y_, linestyle="-", color=blue, linewidth=1.0, label="encrypted")
plt.step(t, y, linestyle="--", color=orange, linewidth=1.0, label="unencrypted")
plt.xlabel("Step")
plt.ylabel(r"$y$")
plt.xlim(0, max(t) + 1)
plt.legend(loc="upper right")

plt.show()
