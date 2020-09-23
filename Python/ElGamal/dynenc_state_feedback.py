#! /usr/bin/env python
# coding: utf-8

# This is a sample code for encrypted state feedback with dynamic-key ElGamal
# encryption. Key pairs and ciphertext of controller parameters are updated
# every sampling period.
# The following is the original paper's information:
# K. Teranishi, N. Shimada, and K. Kogiso, Stability-guaranteed dynamic
# ElGamal cryptosystem for encrypted control systems, IET Control Theory &
# Applications, 2020.

import ElGamal
from Crypto.Util.number import getRandomRange
import numpy as np
import numpy.linalg as la
from control.matlab import *
import matplotlib.pyplot as plt
import figsetup
from colors import *

# sampling time
Ts = 10e-3

# simulation setting
simulation_time = 10
t = np.linspace(0, simulation_time - Ts, int(simulation_time / Ts))

# plant (continuous time)
A = np.array([[1, -1],
              [0,  2]])
B = np.array([[0],
              [1]])
C = np.array([[1, 0],
              [0, 1]])
D = np.array([[0],
              [0]])
sys = c2d(ss(A, B, C, D), Ts)
A = sys.A
B = sys.B
C = sys.C
D = sys.D

# dimension
n = A.shape[0]
m = B.shape[1]
l = C.shape[0]

# controller
Q = np.diag(np.ones(n))
R = np.diag(np.ones(m))
X, _, _ = dare(A, B, Q, R)
F = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)
F = -F

# state
x = np.ones([len(t) + 1, n, 1])
x_ = np.ones([len(t) + 1, n, 1])
x_enc = [[(0, 0)] * n for i in range(len(t) + 1)]

# input
u = np.zeros([len(t), m, 1])
u_ = np.zeros([len(t), m, 1])
u_enc = [[[(0, 0)] * n for i in range(m)] for i in range(len(t))]

# cryptosystem
key_length = 64
csys = ElGamal.DynElGamal(key_length)
csys.KeyGen()

# scaling parameter
gamma_F = 1e8
gamma_x = 1e8

# public key and secret key lists for plot
pk = [csys.h] * (len(t) + 1)
sk = [csys.s] * (len(t) + 1)

# controller encryption
F_enc = [[(0, 0)] * n for i in range(m)]
for i in range(m):
    for j in range(n):
        F_enc[i][j] = csys.Enc(F[i,j], gamma_F)

# simulation w/o encryption
for k in range(len(t)):
    # controller
    u[k,:] = F @ x[k,:]
    # plant update
    x[k+1,:] = A @ x[k,:] + B @ u[k,:]

# simulation w/ encryption
for k in range(len(t)):
    # state encryption
    for i in range(n):
        x_enc[k][i] = csys.Enc(x_[k,i], gamma_x)
    # encrypted controller
    for i in range(m):
        for j in range(n):
            u_enc[k][i][j] = csys.Mult(F_enc[i][j], x_enc[k][j])
    # input decryption
    for i in range(m):
        u_[k,i] = csys.Dec(u_enc[k][i][0], gamma_F * gamma_x)
        for j in range(1, n):
            u_[k,i] += csys.Dec(u_enc[k][i][j], gamma_F * gamma_x)
    # plant update
    x_[k+1,:] = A @ x_[k,:] + B @ u_[k,:]
    # key update
    r = getRandomRange(0, csys.q)
    csys.KeyUpdate(r)
    pk[k+1] = csys.h
    sk[k+1] = csys.s
    # ciphertext update
    for i in range(m):
        for j in range(n):
            F_enc[i][j] = csys.cipherTextUpdate(F_enc[i][j], r)

print("========== PLANT ==========")
print(f"A = {A}")
print(f"B = {B}")
print(f"C = {C}")
print(f"D = {D}")
print(f"sampling period = {Ts} s")
print("========== CONTROLLER ==========")
print(f"F = {F}")
print("========== CRYPTOSYSTEM ==========")
print(f"key length = {key_length}")
print(f"p = {csys.p}")
print(f"q = {csys.q}")
print(f"g = {csys.g}")
print(f"h(0) = {pk[0]}")
print(f"s(0) = {sk[0]}")
print("========== SCALING PARAMETER ==========")
print(f"gamma_F = {gamma_F}")
print(f"gamma_x = {gamma_x}")
print("========== ENCRYPTED CONTROLLER ==========")
print(f"Enc(F) = {F_enc}")

# figure
plt.figure()
plt.plot(t, u[:,0], linestyle='-', color=gray, linewidth=3.0, label='w/o encryption')
plt.plot(t, u_[:,0], linestyle='-', color=blue, linewidth=1.0, label='w/ encryption')
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel(r"$u$")
plt.xlim(0, simulation_time)
plt.ylim(-3.3, 00.5)
plt.legend(loc='lower right')
plt.savefig("./fig/dynenc_state_feedback/input.eps", bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, x[0:-1,0], linestyle='-', color=gray, linewidth=3.0, label='w/o encryption')
plt.plot(t, x[0:-1,1], linestyle='-', color=gray, linewidth=3.0)
plt.plot(t, x_[0:-1,0], linestyle='-', color=blue, linewidth=1.0, label='w/ encryption')
plt.plot(t, x_[0:-1,1], linestyle='-', color=blue, linewidth=1.0)
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel(r"$x$")
plt.xlim(0, simulation_time)
plt.ylim(-0.05, 1.45)
plt.legend(loc='upper right')
plt.savefig("./fig/dynenc_state_feedback/state.eps", bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, [a[0][0] for a in u_enc], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel("Time (s)")
plt.ylabel(r"First element of $\mathsf{Enc}(u)$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/dynenc_state_feedback/enc_input.eps", bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, [a[0][0] for a in x_enc[0:-1]], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel("Time (s)")
plt.ylabel(r"First element of $\mathsf{Enc}(x_1)$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/dynenc_state_feedback/enc_state.eps", bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, pk[0:-1], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$\mathsf{pk}$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/dynenc_state_feedback/public_key.eps", bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, sk[0:-1], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$\mathsf{sk}$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/dynenc_state_feedback/secret_key.eps", bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.show()