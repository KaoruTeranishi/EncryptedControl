#! /usr/bin/env python
# coding: utf-8

# This is a sample code for event-triggered encrypted state feedback with
# dynamic quantizer. The dynamic quantizer guarantees asymptotic stability
# of an encrypted control system even though control input is updated by
# the event triggering mechanism.
# The following is the original paper's information:
# K. Teranishi, J. Ueda, and K. Kogiso, Event-triggered approach to
# increasing sampling period of encrypted control systems, IFAC World
# Congress, 2020.

import ElGamal
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
A = np.array([[ 1,  0,  0],
              [ 0,  0,  1],
              [-1, -2, -1]])
B = np.array([[1, 0],
              [0, 1],
              [0, 0]])
C = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
D = np.array([[0, 0],
              [0, 0],
              [0, 0]])
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
x_bar = np.ones([len(t) + 1, n, 1])
x_enc = [[(0, 0)] * n for i in range(len(t) + 1)]

# input
u = np.zeros([len(t), m, 1])
u_ = np.zeros([len(t), m, 1])
u_enc = [[[(0, 0)] * n for i in range(m)] for i in range(len(t))]

# error
e = np.zeros([len(t), n, 1])
e_ = np.zeros([len(t), n, 1])

# event trigger
et = np.zeros([len(t), 1])
et_ = np.zeros([len(t), 1])

# event time
tk = 0

# cryptosystem
key_length = 20
csys = ElGamal.ElGamal(key_length)
csys.KeyGen()

# maximum width of plaintext space
d_max = csys.dmaxSearch()

# scaling parameter for controller gain
mu_F = 0.01
Q_F = np.diag(np.ones(n))
P_F = dlyap((A + B @ F).T, Q_F)
Omega = 2 * (-la.norm((A + B @ F).T @ P_F @ B) + np.sqrt(la.norm((A + B @ F).T @ P_F @ B) ** 2 + min(la.eig(Q_F)[0]) * la.norm(B.T @ P_F @ B))) / (np.sqrt(m * n) * la.norm(B.T @ P_F @ B))
gamma_F = d_max / Omega + mu_F

# quantized controller gain used in design of scaling parameter for state
F_bar = F
for i in range(m):
    for j in range(n):
        F_bar[i,j] = csys.Decode(csys.Encode(F[i,j], gamma_F), gamma_F)

# scaling parameter for state
mu_x = 0.01
Q_x = np.diag(np.ones(n))
P_x = dlyap((A + B @ F_bar).T, Q_F)
Theta = np.sqrt(n) * (la.norm((A + B @ F_bar).T @ P_x @ B @ F_bar) + np.sqrt(la.norm((A + B @ F_bar).T @ P_x @ B @ F_bar) ** 2 + min(la.eig(Q_x)[0]) * la.norm(F_bar.T @ B.T @ P_x @ B @ F_bar))) / (2 * min(la.eig(Q_x)[0]))
gamma_x = [0] * len(t)

# event-triggering parameter
Q_e = np.diag(np.ones(n))
P_e = dlyap((A + B @ F).T, Q_e)
sigma = (la.norm((A + B @ F).T @ P_e @ B @ F) + np.sqrt(la.norm((A + B @ F).T @ P_e @ B @ F) ** 2 + min(la.eig(Q_e)[0]) * la.norm(F.T @ B.T @ P_e @ B @ F))) / min(la.eig(Q_e)[0])

Q_e = np.diag(np.ones(n))
P_e = dlyap((A + B @ F_bar).T, Q_e)
sigma_ = (la.norm((A + B @ F_bar).T @ P_e @ B @ F_bar) + np.sqrt(la.norm((A + B @ F_bar).T @ P_e @ B @ F_bar) ** 2 + min(la.eig(Q_e)[0]) * la.norm(F_bar.T @ B.T @ P_e @ B @ F_bar))) / min(la.eig(Q_e)[0])

# controller encryption
F_enc = [[(0, 0)] * n for i in range(m)]
for i in range(m):
    for j in range(n):
        F_enc[i][j] = csys.Enc(F[i,j], gamma_F)

# simulation w/o encryption
for k in range(len(t)):
    # error
    e[k,:] = x[tk,:] - x[k,:]
    # event triggering
    if k == 0 or la.norm(x[k,:]) <= sigma * la.norm(e[k,:]):
        u[k,:] = F @ x[k,:]
        tk = k
        et[k] = 1
    else:
        u[k,:] = u[tk,:]
    # plant update
    x[k+1,:] = A @ x[k,:] + B @ u[k,:]

# simulation w/ encryption
for k in range(len(t)):
    # error
    e_[k,:] = x_bar[tk,:] - x_[k,:]
    # event triggering
    if k == 0 or la.norm(x_[k,:]) <= sigma_ * la.norm(e_[k,:]):
        # scaling parameter update
        gamma_x[k] = Theta * d_max / la.norm(x_[k]) + mu_x
        # state encryption
        for i in range(n):
            x_bar[k][i] = csys.Decode(csys.Encode(x_[k,i], gamma_x[k]), gamma_x[k])
            x_enc[k][i] = csys.Enc(x_[k,i], gamma_x[k])
        # encrypted controller
        for i in range(m):
            for j in range(n):
                u_enc[k][i][j] = csys.Mult(F_enc[i][j], x_enc[k][j])
        # input decryption
        for i in range(m):
            u_[k,i] = csys.Dec(u_enc[k][i][0], gamma_F * gamma_x[k])
            for j in range(1, n):
                u_[k,i] += csys.Dec(u_enc[k][i][j], gamma_F * gamma_x[k])
        tk = k
        et_[k] = 1
    else:
        u_[k,:] = u_[tk,:]
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
print("========== CRYPTOSYSTEM ==========")
print(f"key length = {key_length}")
print(f"p = {csys.p}")
print(f"q = {csys.q}")
print(f"g = {csys.g}")
print(f"h = {csys.h}")
print(f"s = {csys.s}")
print(f"dmax = {d_max}")
print("========== SCALING PARAMETER ==========")
print(f"Omega = {Omega}")
print(f"Theta = {Theta}")
print(f"gamma_F = {gamma_F}")
print(f"gamma_x(0) = {gamma_x[0]}")
print("========== TRIGGERING MECHANISM ==========")
print(f"sigma (unencrypted) = {sigma}")
print(f"sigma (encrypted) = {sigma_}")
print("========== ENCRYPTED CONTROLLER ==========")
print(f"Enc(F) = {F_enc}")

# figure
plt.figure()
plt.plot(t, u[:,0], linestyle='-', color=gray, linewidth=3.0, label='w/o encryption')
plt.plot(t, u[:,1], linestyle='-', color=gray, linewidth=3.0)
plt.plot(t, u_[:,0], linestyle='-', color=blue, linewidth=1.0, label='w/ encryption')
plt.plot(t, u_[:,1], linestyle='-', color=blue, linewidth=1.0)
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel(r"$u$")
plt.xlim(0, simulation_time)
plt.ylim(-3, 0.5)
plt.legend(loc='lower right')
plt.savefig("./fig/evt_enc_state_feedback/input.eps", bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, x[0:-1,0], linestyle='-', color=gray, linewidth=3.0, label='w/o encryption')
plt.plot(t, x[0:-1,1], linestyle='-', color=gray, linewidth=3.0)
plt.plot(t, x[0:-1,2], linestyle='-', color=gray, linewidth=3.0)
plt.plot(t, x_[0:-1,0], linestyle='-', color=blue, linewidth=1.0, label='w/ encryption')
plt.plot(t, x_[0:-1,1], linestyle='-', color=blue, linewidth=1.0)
plt.plot(t, x_[0:-1,2], linestyle='-', color=blue, linewidth=1.0)
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=0.5)
plt.xlabel("Time (s)")
plt.ylabel(r"$x$")
plt.xlim(0, simulation_time)
plt.ylim(-0.6, 1.1)
plt.legend(loc='upper right')
plt.savefig("./fig/evt_enc_state_feedback/state.eps", bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
markers, lines, base = plt.stem([t[i] for i in np.where(et == 1)][0][:], et[et == 1], basefmt=' ')
plt.setp(markers, color=blue, markersize=3.0)
plt.setp(lines, linestyle='-', color=blue, linewidth=1.0)
plt.xlabel("Time (s)")
plt.ylabel("Event")
plt.xlim(0, simulation_time)
plt.ylim(0, 1.05)
plt.yticks([0, 1])
plt.savefig("./fig/evt_enc_state_feedback/trigger.eps", bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
markers, lines, base = plt.stem([t[i] for i in np.where(et_ == 1)][0][:], et_[et_ == 1], basefmt=' ')
plt.setp(markers, color=blue, markersize=3.0)
plt.setp(lines, linestyle='-', color=blue, linewidth=1.0)
plt.xlabel("Time (s)")
plt.ylabel("Event")
plt.xlim(0, simulation_time)
plt.ylim(0, 1.05)
plt.yticks([0, 1])
plt.savefig("./fig/evt_enc_state_feedback/enc_trigger.eps", bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.show()