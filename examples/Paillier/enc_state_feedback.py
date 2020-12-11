#! /usr/bin/env python

# This is a sample code for encrypted state feedback.
# The following is the original paper's information (but it uses output
# feedback):
# F. Farokhi, I. Shames, and N. Batterham, Secure and private control using
# semi-homomorphic encryption, Control Engineering Practice, vol. 67,
# pp. 13–20, 2017.

import eclib.paillier as paillier
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
x_enc = [[0 for j in range(n)] for i in range(len(t) + 1)]

# input
u = np.zeros([len(t), m])
u_ = np.zeros([len(t), m])
u_enc = [[0 for j in range(m)] for i in range(len(t))]

# cryptosystem
key_length = 64
params, pk, sk = paillier.keygen(key_length)

# scaling parameter
a1 = 9
b1 = 7
a2 = 9
b2 = 8
a = a1 + a2 + n
b = b1 + b2

# controller encoding
F_ecd = paillier.encode(F, a, b, a1, b1)

# simulation w/o encryption
for k in range(len(t)):
    # controller
    u[k,:] = F @ x[k,:]
    # plant update
    x[k+1,:] = A @ x[k,:] + B @ u[k,:]

# simulation w/ encryption
for k in range(len(t)):
    # state encryption
    x_enc[k] = paillier.enc(params, pk, x_[k], a, b, a2, b2)
    # encrypted controller
    u_enc[k] = paillier.mult(params, F_ecd, x_enc[k])
    # input decryption
    u_[k] = paillier.dec(params, sk, u_enc[k], a, b)
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
print('========== paillier parameter ==========')
print(f'key length = {key_length} bit')
print(f'n = {params.n}')
print(f'n^2 = {params.n_square}')
print(f'g = {pk}')
print(f'lambda = {sk.lmd}')
print(f'mu = {sk.mu}')
print('========== encoding parameter ==========')
print(f'a1 = {a1}')
print(f'b1 = {b1}')
print(f'a2 = {a2}')
print(f'b2 = {b2}')
print(f'a = {a}')
print(f'b = {b}')
print('========== encoded controller ==========')
print(f'Ecd(F) = {F_ecd}')

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
plt.savefig('../fig/Paillier/enc_state_feedback/input.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

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
plt.savefig('../fig/Paillier/enc_state_feedback/state.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, u_enc, linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post')
plt.xlabel('Time (s)')
plt.ylabel(r'$\mathsf{Enc}(u)$')
plt.xlim(0, simulation_time)
plt.savefig('../fig/Paillier/enc_state_feedback/enc_input.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, x_enc[0:-1], linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post')
plt.xlabel('Time (s)')
plt.ylabel(r'$\mathsf{Enc}(x_1)$')
plt.xlim(0, simulation_time)
plt.savefig('../fig/Paillier/enc_state_feedback/enc_state.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.show()
