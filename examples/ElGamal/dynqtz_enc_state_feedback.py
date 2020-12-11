#! /usr/bin/env python

# This is a sample code for encrypted state feedback with dynamic quantizer.
# The dynamic quantizer guarantees asymptotic stability of an encrypted
# control system.
# The following is the original paper's information:
# K. Teranishi, N. Shimada, and K. Kogiso, Stability analysis and dynamic
# quantizer for controller encryption, IEEE Conference on Decision and
# Control, pp. 7184-7189, 2019.

import eclib.elgamal as elgamal
from eclib.colors import *
import eclib.figsetup
import numpy as np
import numpy.linalg as la
from control.matlab import *
import matplotlib.pyplot as plt

# sampling time
Ts = 10e-3

# simulation setting
simulation_time = 20
t = np.linspace(0, simulation_time, int(simulation_time / Ts))

# plant (continuous time)
A = np.array([[1, -1],
              [0,  2]])
B = np.array([[0],
              [1]])
C = np.array([[1, 0],
              [0, 1]])
D = np.array([[0],
              [0]])

# discretization of plant
sys = c2d(ss(A, B, C, D), Ts)
A = sys.A
B = sys.B
C = sys.C
D = sys.D

# dimension of plant
n = A.shape[0]
m = B.shape[1]
l = C.shape[0]

# controller
Q = np.identity(n)
R = np.identity(m)
X, _, _ = dare(A, B, Q, R)
F = -la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

# state
x = np.ones([len(t) + 1, n])
x_ = np.ones([len(t) + 1, n])
x_enc = [[[0, 0] for j in range(n)] for i in range(len(t) + 1)]

# input
u = np.zeros([len(t), m])
u_ = np.zeros([len(t), m])
u_enc = [[[[0, 0] for k in range(n)] for j in range(m)] for i in range(len(t))]

# cryptosystem
key_length = 20
params, pk, sk = elgamal.keygen(key_length)

# maximum width of plaintext space
dmax = elgamal.get_dmax(params)

# resolution of quantizer for controller gain
mu_F = 0.99
Q_F = np.identity(n)
P_F = dlyap((A + B @ F).T, Q_F)
Omega = 2 * (-la.norm((A + B @ F).T @ P_F @ B) + np.sqrt(la.norm((A + B @ F).T @ P_F @ B) ** 2 + min(la.eig(Q_F)[0]) * la.norm(B.T @ P_F @ B))) / (np.sqrt(m * n) * la.norm(B.T @ P_F @ B))
delta_F = Omega / dmax * mu_F

# quantized controller gain used in design of resolution for state
F_bar = np.array(elgamal.decode(params, elgamal.encode(params, F, delta_F), delta_F))

# resolution of quantizer for state
mu_x = 0.99
Q_x = np.identity(n)
P_x = dlyap((A + B @ F_bar).T, Q_x)
Theta = np.sqrt(n) * (la.norm((A + B @ F_bar).T @ P_x @ B @ F_bar) + np.sqrt(la.norm((A + B @ F_bar).T @ P_x @ B @ F_bar) ** 2 + min(la.eig(Q_x)[0]) * la.norm(F_bar.T @ B.T @ P_x @ B @ F_bar))) / (2 * min(la.eig(Q_x)[0]))
delta_x = [0] * len(t)

# controller encryption
F_enc = elgamal.enc(params, pk, F, delta_F)

# simulation w/o encryption
for k in range(len(t)):
    # controller
    u[k,:] = F @ x[k,:]
    # plant update
    x[k+1,:] = A @ x[k,:] + B @ u[k,:]

# simulation w/ encryption
for k in range(len(t)):
    # resolution update
    delta_x[k] = la.norm(x_[k]) / (Theta * dmax) * mu_x
    # state encryption
    x_enc[k] = elgamal.enc(params, pk, x_[k], delta_x[k])
    # encrypted controller
    u_enc[k] = elgamal.mult(params, F_enc, x_enc[k])
    # input decryption
    u_[k] = elgamal.dec_add(params, sk, u_enc[k], delta_F * delta_x[k])
    # plant update
    x_[k+1,:] = A @ x_[k,:] + B @ u_[k,:]

np.set_printoptions(formatter={'float': '{:.2f}'.format})
print('========== plant ==========')
print(f'A = \n{A}')
print(f'B = \n{B}')
print(f'C = \n{C}')
print(f'D = \n{D}')
print(f'sampling time = {Ts} s')
print('========== controller ==========')
print(f'F = {F}')
print('========== ElGamal parameter ==========')
print(f'key length = {key_length} bit')
print(f'p = {params.p}')
print(f'q = {params.q}')
print(f'g = {params.g}')
print(f'h = {pk}')
print(f's = {sk}')
print(f'dmax = {dmax}')
print('========== resolution ==========')
print(f'Omega = {Omega}')
print(f'Theta = {Theta}')
print(f'delta_F = {delta_F}')
print(f'delta_x(0) = {delta_x[0]}')
print('========== encrypted controller ==========')
print(f'Enc(F) = {F_enc}')

# figure
plt.figure()
plt.plot(t, u, linestyle='-', color=gray, linewidth=3.0, drawstyle='steps-post', label='unencrypted')
plt.plot(t, u_, linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post', label='encrypted')
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel(r'$u$')
plt.xlim(0, simulation_time)
plt.ylim(-3.3, 0.5)
plt.legend(loc='lower right')
plt.savefig('../fig/ElGamal/dynqtz_enc_state_feedback/input.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, x[0:-1,0], linestyle='-', color=gray, linewidth=3.0, drawstyle='steps-post', label='unencrypted')
plt.plot(t, x[0:-1,1], linestyle='-', color=gray, linewidth=3.0, drawstyle='steps-post')
plt.plot(t, x_[0:-1,0], linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post', label='encrypted')
plt.plot(t, x_[0:-1,1], linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post')
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel(r'$x$')
plt.xlim(0, simulation_time)
plt.ylim(-0.05, 1.5)
plt.legend(loc='upper right')
plt.savefig('../fig/ElGamal/dynqtz_enc_state_feedback/state.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, [la.norm(a) for a in x[0:-1] - x_[0:-1]], linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post')
plt.xlabel('Time (s)')
plt.ylabel(r'$\|e\|$')
plt.xlim(0, simulation_time)
plt.ylim(0, 0.015)
plt.savefig('../fig/ElGamal/dynqtz_enc_state_feedback/quantization_error.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, delta_x, linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel(r'$\Delta_{x}$')
plt.xlim(0, simulation_time)
plt.ylim(1e-15, 1e0)
plt.yticks([1e0, 1e-5, 1e-10, 1e-15])
plt.savefig('../fig/ElGamal/dynqtz_enc_state_feedback/resolution.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.show()
