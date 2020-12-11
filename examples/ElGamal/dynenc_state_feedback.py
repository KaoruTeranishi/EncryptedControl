#! /usr/bin/env python

# This is a sample code for encrypted state feedback with dynamic-key ElGamal
# encryption. Key pairs and ciphertext of controller parameters are updated
# every sampling period.
# The following is the original paper's information:
# K. Teranishi, N. Shimada, and K. Kogiso, Stability-guaranteed dynamic
# ElGamal cryptosystem for encrypted control systems, IET Control Theory &
# Applications, vol. 14, iss. 16, pp. 2242-2252, 2020.

import eclib.elgamal as elgamal
from eclib.colors import *
import eclib.figsetup
from eclib.randutils import get_rand
import numpy as np
import numpy.linalg as la
from control.matlab import *
import matplotlib.pyplot as plt

# sampling time
Ts = 10e-3

# simulation setting
simulation_time = 10
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
pk = [0] * (len(t) + 1)
sk = [0] * (len(t) + 1)
key_length = 64
params, pk[0], sk[0] = elgamal.keygen(key_length)

# resolution
delta_F = 1e-8
delta_x = 1e-8

# controller encryption
F_enc = elgamal.enc(params, pk[0], F, delta_F)

# simulation w/o encryption
for k in range(len(t)):
    # controller
    u[k,:] = F @ x[k,:]
    # plant update
    x[k+1,:] = A @ x[k,:] + B @ u[k,:]

# simulation w/ encryption
for k in range(len(t)):
    # state encryption
    x_enc[k] = elgamal.enc(params, pk[k], x_[k], delta_x)
    # encrypted controller
    u_enc[k] = elgamal.mult(params, F_enc, x_enc[k])
    # input decryption
    u_[k] = elgamal.dec_add(params, sk[k], u_enc[k], delta_F * delta_x)
    # plant update
    x_[k+1,:] = A @ x_[k,:] + B @ u_[k,:]
    # key update
    r = get_rand(1, params.q)
    pk[k+1], sk[k+1] = elgamal.update_key(params, pk[k], sk[k], r)
    # ciphertext update
    F_enc = elgamal.update_ctxt(params, F_enc, r)

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
print(f'h(0) = {pk[0]}')
print(f's(0) = {sk[0]}')
print('========== resolution ==========')
print(f'delta_F = {delta_F}')
print(f'delta_x = {delta_x}')
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
plt.savefig('../fig/ElGamal/dynenc_state_feedback/input.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, x[0:-1,0], linestyle='-', color=gray, linewidth=3.0, drawstyle='steps-post', label='unencrypted')
plt.plot(t, x[0:-1,1], linestyle='-', color=gray, linewidth=3.0, drawstyle='steps-post')
plt.plot(t, x_[0:-1,0], linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post', label='encrypted')
plt.plot(t, x_[0:-1,1], linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post')
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel(r'$x$')
plt.xlim(0, simulation_time)
plt.ylim(-0.05, 1.45)
plt.legend(loc='upper right')
plt.savefig('../fig/ElGamal/dynenc_state_feedback/state.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, pk[0:-1], linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post')
plt.xlabel('Time (s)')
plt.ylabel(r'$\mathsf{pk}$')
plt.xlim(0, simulation_time)
plt.savefig('../fig/ElGamal/dynenc_state_feedback/public_key.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, sk[0:-1], linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post')
plt.xlabel('Time (s)')
plt.ylabel(r'$\mathsf{sk}$')
plt.xlim(0, simulation_time)
plt.savefig('../fig/ElGamal/dynenc_state_feedback/secret_key.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.show()
