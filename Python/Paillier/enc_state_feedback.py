#! /usr/bin/env python
# coding: utf-8

# This is a sample code for encrypted state feedback with a quantizer proposed
# in the following paper:
# F. Farokhi, I. Shames, and N. Batterham, Secure and private control using
# semi-homomorphic encryption, Control Engineering Practice, vol. 67,
# pp. 13–20, 2017.

import Paillier
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
csys = Paillier.Paillier(key_length)
csys.KeyGen()

# fixed point number parameter for quantization
a1 = 9
b1 = 7
a2 = 9
b2 = 8
a = a1 + a2 + n
b = b1 + b2

# controller encryption
F_pt = [[0] * n for i in range(m)]
for i in range(m):
    for j in range(n):
        F_pt[i][j] = csys.Encode(F[i,j], a, b, a1, b1)

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
        x_enc[k][i] = csys.Enc(x_[k,i], a, b, a2, b2)
    # encrypted controller
    for i in range(m):
        u_enc[k][i] = csys.Mult(x_enc[k][0], F_pt[i][0])
        for j in range(1, n):
            u_enc[k][i] = csys.Add(u_enc[k][i], csys.Mult(x_enc[k][j], F_pt[i][j]))
    # input decryption
    for i in range(m):
        u_[k,i] = csys.Dec(u_enc[k][i], a, b)
    # plant update
    x_[k+1,:] = A @ x_[k,:] + B @ u_[k,:]

print("========== PLANT ==========")
print(f"A = {A}")
print(f"B = {B}")
print(f"C = {C}")
print(f"D = {D}")
print(f"sampling period = {Ts} s")
print("========== CONTROLLER ==========")
print(f"F = {F}")
print(f"Plaintext of F = {F_pt}")
print("========== CRYPTOSYSTEM ==========")
print(f"key length = {key_length}")
print(f"p = {csys.p}")
print(f"q = {csys.q}")
print(f"n = {csys.n}")
print(f"g = {csys.g}")
print(f"lambda = {csys.lmd}")
print(f"mu = {csys.mu}")

# figure
plt.figure()
plt.plot(t, u[:,0], linestyle='-', color=gray, linewidth=3.0, label='w/o encryption')
plt.plot(t, u_[:,0], linestyle='-', color=blue, linewidth=1.0, label='w/ encryption')
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel(r"$u$")
plt.xlim(0, simulation_time)
plt.ylim(-3.3, 0.5)
plt.legend(loc='lower right')
plt.savefig("./fig/enc_state_feedback/input.eps", bbox_inches='tight', pad_inches=0.05, transparent=True)

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
plt.savefig("./fig/enc_state_feedback/state.eps", bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, [a[0] for a in u_enc], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$\mathsf{Enc}(u)$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/enc_state_feedback/enc_input.eps", bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, [a[0] for a in x_enc[0:-1]], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$\mathsf{Enc}(x_1)$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/enc_state_feedback/enc_state.eps", bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.show()