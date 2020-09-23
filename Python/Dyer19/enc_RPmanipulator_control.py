#! /usr/bin/env python
# coding: utf-8

# This is a sample code for encrypted state feedback.
# The following is the original paper's information:
# K. Teranishi, K. Kogiso, and J. Ueda, Encrypted feedback linearization and
# motion control for manipulator with somewhat homomorphic encryption,
# IEEE/ASME International Conference on Advanced Intelligent Mechatronics,
# pp.613-618, 2020.

import Dyer19
from manipulator import *
import math
import numpy as np
import numpy.linalg as la
from control.matlab import *
import matplotlib.pyplot as plt
import figsetup
from colors import *
from tqdm import trange

# sampling time
dt = 10e-3

# simulation setting
simulation_time = 3
t = np.linspace(0, simulation_time - dt, int(simulation_time / dt))

# RP manipulator
m = [3.0, 3.0]
l = [1.0, 1.0]
I = [1e-3, 1e-3]
g = 9.81
q_init = [.0, 1.0]
q_dot_init = [.0, .0]
plant = RPManipulator(m, l, I, q_init, q_dot_init, dt)
plant_ = RPManipulator(m, l, I, q_init, q_dot_init, dt)

# signal
r_th = np.zeros(len(t))
th = np.zeros(len(t))
th_dot = np.zeros(len(t))
e_th = np.zeros(len(t))
e_th_dot = np.zeros(len(t))
r_d = np.zeros(len(t))
d = np.zeros(len(t))
d_dot = np.zeros(len(t))
e_d = np.zeros(len(t))
e_d_dot = np.zeros(len(t))
tau = np.zeros(len(t))
f = np.zeros(len(t))

r_th_ = np.zeros(len(t))
th_ = np.zeros(len(t))
th_dot_ = np.zeros(len(t))
e_th_ = np.zeros(len(t))
e_th_dot_ = np.zeros(len(t))
r_d_ = np.zeros(len(t))
d_ = np.zeros(len(t))
d_dot_ = np.zeros(len(t))
e_d_ = np.zeros(len(t))
e_d_dot_ = np.zeros(len(t))
tau_ = np.zeros(len(t))
f_ = np.zeros(len(t))

# controller input
xi_th = np.zeros([len(t), 11, 1])
xi_d = np.zeros([len(t), 6, 1])

xi_th_ = np.zeros([len(t), 11, 1])
xi_d_ = np.zeros([len(t), 6, 1])

xi_th_enc = [[0] * 11 for i in range(len(t))]
xi_d_enc = [[0] * 6 for i in range(len(t))]

# controller output
psi_th = np.zeros([len(t), 3, 1])
psi_d = np.zeros([len(t), 3, 1])

psi_th_ = np.zeros([len(t), 3, 1])
psi_d_ = np.zeros([len(t), 3, 1])

psi_th_enc = [[0] * 3 for i in range(len(t))]
psi_d_enc = [[0] * 3 for i in range(len(t))]

# controller parameter
Kp_th = 4
Kd_th = 12
Kp_d = 40
Kd_d = 12
gpd = 100
a = 2 * gpd / (2 + dt * gpd)
b = (2 - dt * gpd) / (2 + dt * gpd)
A_th = np.array([[0, 0], [-a, b]])
B_th = np.array([[1, -1, 0, 0, 0, 0, 0, 0, 0], [a, -a, 0, 0, 0, 0, 0, 0, 0]])
C_th = np.array([-(m[0] * pow(l[0], 2) + I[0] + I[1]) * Kd_th * a, (m[0] * pow(l[0], 2) + I[0] + I[1]) * Kd_th * b])
D_th = np.array([(m[0] * pow(l[0], 2) + I[0] + I[1]) * (Kp_th + Kd_th * a), -(m[0] * pow(l[0], 2) + I[0] + I[1]) * (Kp_th + Kd_th * a), -m[1] * Kd_th * a, m[1] * Kd_th * b, m[1] * (Kp_th + Kd_th * a), -m[1] * (Kp_th + Kd_th * a), 2 * m[1], m[0] * l[0] * g, m[1] * g])
Phi_th = np.block([[A_th, B_th], [C_th, D_th]])
A_d = np.array([[0, 0], [-a, b]])
B_d = np.array([[1, -1, 0, 0], [a, -a, 0, 0]])
C_d = np.array([-m[1] * Kd_d * a, m[1] * Kd_d * b])
D_d = np.array([m[1] * (Kp_d + Kd_d * a), -m[1] * (Kp_d + Kd_d * a), -m[1], m[1] * g])
Phi_d = np.block([[A_d, B_d], [C_d, D_d]])

# cryptosystem
csys = Dyer19.HE1N(256, 1, 62)
csys.KeyGen()
csys.Pgen()

# scaling parameter
gamma = 1.0 / 1e-4

# controller encryption
Phi_th_enc = [[0] * 11 for i in range(3)]
for i in range(3):
    for j in range(11):
        Phi_th_enc[i][j] = csys.Enc(Phi_th[i,j], gamma)
Phi_d_enc = [[0] * 6 for i in range(3)]
for i in range(3):
    for j in range(6):
        Phi_d_enc[i][j] = csys.Enc(Phi_d[i,j], gamma)

# simulation w/o encryption
for k in trange(len(t)):
    # reference
    r_th[k] = np.pi / 6
    r_d[k] = 1.5

    # measurement
    th[k] = plant.q[0]
    th_dot[k] = plant.q_dot[0]
    d[k] = plant.q[1]
    d_dot[k] = plant.q_dot[1]

    # signal composition
    xi_th[k,0] = e_th[k-1]
    xi_th[k,1] = e_th_dot[k-1]
    xi_th[k,2] = r_th[k]
    xi_th[k,3] = th[k]
    xi_th[k,4] = e_th[k-1] * pow(d[k], 2)
    xi_th[k,5] = e_th_dot[k-1] * pow(d[k], 2)
    xi_th[k,6] = r_th[k] * pow(d[k], 2)
    xi_th[k,7] = th[k] * pow(d[k], 2)
    xi_th[k,8] = d[k] * d_dot[k] * th_dot[k]
    xi_th[k,9] = math.cos(th[k])
    xi_th[k,10] = d[k] * math.cos(th[k])
    xi_d[k,0] = e_d[k-1]
    xi_d[k,1] = e_d_dot[k-1]
    xi_d[k,2] = r_d[k]
    xi_d[k,3] = d[k]
    xi_d[k,4] = d[k] * pow(th_dot[k], 2)
    xi_d[k,5] = math.sin(th[k])

    # controller
    psi_th[k,:] = Phi_th @ xi_th[k,:]
    psi_d[k,:] = Phi_d @ xi_d[k,:]

    # signal decomposition
    e_th[k] = psi_th[k,0]
    e_th_dot[k] = psi_th[k,1]
    tau[k] = psi_th[k,2]
    e_d[k] = psi_d[k,0]
    e_d_dot[k] = psi_d[k,1]
    f[k] = psi_d[k,2]

    # plant update
    plant.update([tau[k], f[k]])

# simulation w/ encryption
for k in trange(len(t)):
    # reference
    r_th_[k] = np.pi / 6
    r_d_[k] = 1.5

    # measurement
    th_[k] = plant_.q[0]
    th_dot_[k] = plant_.q_dot[0]
    d_[k] = plant_.q[1]
    d_dot_[k] = plant_.q_dot[1]

    # signal composition
    xi_th_[k,0] = e_th_[k-1]
    xi_th_[k,1] = e_th_dot_[k-1]
    xi_th_[k,2] = r_th_[k]
    xi_th_[k,3] = th_[k]
    xi_th_[k,4] = e_th_[k-1] * pow(d_[k], 2)
    xi_th_[k,5] = e_th_dot_[k-1] * pow(d_[k], 2)
    xi_th_[k,6] = r_th_[k] * pow(d_[k], 2)
    xi_th_[k,7] = th_[k] * pow(d_[k], 2)
    xi_th_[k,8] = d_[k] * d_dot_[k] * th_dot_[k]
    xi_th_[k,9] = math.cos(th_[k])
    xi_th_[k,10] = d_[k] * math.cos(th_[k])
    xi_d_[k,0] = e_d_[k-1]
    xi_d_[k,1] = e_d_dot_[k-1]
    xi_d_[k,2] = r_d_[k]
    xi_d_[k,3] = d_[k]
    xi_d_[k,4] = d_[k] * pow(th_dot_[k], 2)
    xi_d_[k,5] = math.sin(th_[k])

    # signal encryption
    for i in range(11):
        xi_th_enc[k][i] = csys.Enc(xi_th_[k,i], gamma)
    for i in range(6):
        xi_d_enc[k][i] = csys.Enc(xi_d_[k,i], gamma)

    # encrypted controller
    for i in range(3):
        psi_th_enc[k][i] = csys.Mult(Phi_th_enc[i][0], xi_th_enc[k][0])
        for j in range(1, 11):
            psi_th_enc[k][i] = csys.Add(psi_th_enc[k][i], csys.Mult(Phi_th_enc[i][j], xi_th_enc[k][j]))
    for i in range(3):
        psi_d_enc[k][i] = csys.Mult(Phi_d_enc[i][0], xi_d_enc[k][0])
        for j in range(1, 6):
            psi_d_enc[k][i] = csys.Add(psi_d_enc[k][i], csys.Mult(Phi_d_enc[i][j], xi_d_enc[k][j]))

    # signal decryption
    for i in range(3):
        psi_th_[k,i] = csys.Dec(psi_th_enc[k][i], gamma ** 2)
    for i in range(3):
        psi_d_[k,i] = csys.Dec(psi_d_enc[k][i], gamma ** 2)

    # signal decomposition
    e_th_[k] = psi_th_[k,0]
    e_th_dot_[k] = psi_th_[k,1]
    tau_[k] = psi_th_[k,2]
    e_d_[k] = psi_d_[k,0]
    e_d_dot_[k] = psi_d_[k,1]
    f_[k] = psi_d_[k,2]

    # plant update
    plant_.update([tau_[k], f_[k]])

# figure
plt.figure()
plt.plot(t, th, linestyle='-', color=gray, linewidth=3.0, label='w/o encryption')
plt.plot(t, th_, linestyle='-', color=blue, linewidth=1.0, label='w/ encryption')
plt.plot(t, r_th_, linestyle='--', color=black, linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel(r"$\theta_{1}$")
plt.xlim(0, simulation_time)
plt.ylim(0, 0.6)
plt.xticks([0, 1, 2, 3], ('0', '1', '2', '3'))
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
plt.legend(loc='lower right')
plt.savefig("./fig/enc_RPmanipulator_control/theta.eps", bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, d, linestyle='-', color=gray, linewidth=3.0, label='w/o encryption')
plt.plot(t, d_, linestyle='-', color=blue, linewidth=1.0, label='w/ encryption')
plt.plot(t, r_d_, linestyle='--', color=black, linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel(r"$d_{2}$")
plt.xlim(0, simulation_time)
plt.ylim(1, 1.6)
plt.xticks([0, 1, 2, 3], ('0', '1', '2', '3'))
plt.yticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
plt.legend(loc='lower right')
plt.savefig("./fig/enc_RPmanipulator_control/d.eps", bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.show()