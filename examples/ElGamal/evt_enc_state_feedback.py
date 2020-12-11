#! /usr/bin/env python

# This is a sample code for event-triggered encrypted state feedback with
# dynamic quantizer. The dynamic quantizer guarantees asymptotic stability
# of an encrypted control system even though control input is updated by
# the event triggering mechanism.
# The following is the original paper's information:
# K. Teranishi, J. Ueda, and K. Kogiso, Event-triggered approach to
# increasing sampling period of encrypted control systems, IFAC World
# Congress, 2020.

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
x_bar = np.zeros([len(t) + 1, n])
x_enc = [[[0, 0] for j in range(n)] for i in range(len(t) + 1)]

# input
u = np.zeros([len(t), m])
u_ = np.zeros([len(t), m])
u_enc = [[[[0, 0] for k in range(n)] for j in range(m)] for i in range(len(t))]

# error
e = np.zeros([len(t), n])
e_ = np.zeros([len(t), n])

# event trigger
et = np.zeros(len(t))
et_ = np.zeros(len(t))

# event time
tk = 0

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

# scaling parameter for state
mu_x = 0.99
Q_x = np.identity(n)
P_x = dlyap((A + B @ F_bar).T, Q_x)
Theta = np.sqrt(n) * (la.norm((A + B @ F_bar).T @ P_x @ B @ F_bar) + np.sqrt(la.norm((A + B @ F_bar).T @ P_x @ B @ F_bar) ** 2 + min(la.eig(Q_x)[0]) * la.norm(F_bar.T @ B.T @ P_x @ B @ F_bar))) / (2 * min(la.eig(Q_x)[0]))
delta_x = [0] * len(t)

# event-triggering parameter
Q_e = np.identity(n)
P_e = dlyap((A + B @ F).T, Q_e)
sigma = (la.norm((A + B @ F).T @ P_e @ B @ F) + np.sqrt(la.norm((A + B @ F).T @ P_e @ B @ F) ** 2 + min(la.eig(Q_e)[0]) * la.norm(F.T @ B.T @ P_e @ B @ F))) / min(la.eig(Q_e)[0])

Q_e = np.identity(n)
P_e = dlyap((A + B @ F_bar).T, Q_e)
sigma_ = (la.norm((A + B @ F_bar).T @ P_e @ B @ F_bar) + np.sqrt(la.norm((A + B @ F_bar).T @ P_e @ B @ F_bar) ** 2 + min(la.eig(Q_e)[0]) * la.norm(F_bar.T @ B.T @ P_e @ B @ F_bar))) / min(la.eig(Q_e)[0])

# controller encryption
F_enc = elgamal.enc(params, pk, F, delta_F)

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
        delta_x[k] = la.norm(x_[k]) / (Theta * dmax) * mu_x
        # state encryption
        x_bar[k] = elgamal.decode(params, elgamal.encode(params, x_[k], delta_x[k]), delta_x[k])
        x_enc[k] = elgamal.enc(params, pk, x_[k], delta_x[k])
        # encrypted controller
        u_enc[k] = elgamal.mult(params, F_enc, x_enc[k])
        # input decryption
        u_[k] = elgamal.dec_add(params, sk, u_enc[k], delta_F * delta_x[k])
        tk = k
        et_[k] = 1
    else:
        u_[k,:] = u_[tk,:]
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
print('========== triggering mechanism ==========')
print(f'sigma (unencrypted) = {sigma}')
print(f'sigma (encrypted) = {sigma_}')
print('========== encrypted controller ==========')
print(f'Enc(F) = {F_enc}')

# figure
plt.figure()
plt.plot(t, u[:,0], linestyle='-', color=gray, linewidth=3.0, drawstyle='steps-post', label='unencrypted')
plt.plot(t, u[:,1], linestyle='-', color=gray, linewidth=3.0, drawstyle='steps-post')
plt.plot(t, u_[:,0], linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post', label='encrypted')
plt.plot(t, u_[:,1], linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post')
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel(r'$u$')
plt.xlim(0, simulation_time)
plt.ylim(-3, 0.5)
plt.legend(loc='lower right')
plt.savefig('../fig/ElGamal/evt_enc_state_feedback/input.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, x[0:-1,0], linestyle='-', color=gray, linewidth=3.0, drawstyle='steps-post', label='unencrypted')
plt.plot(t, x[0:-1,1], linestyle='-', color=gray, linewidth=3.0, drawstyle='steps-post')
plt.plot(t, x[0:-1,2], linestyle='-', color=gray, linewidth=3.0, drawstyle='steps-post')
plt.plot(t, x_[0:-1,0], linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post', label='encrypted')
plt.plot(t, x_[0:-1,1], linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post')
plt.plot(t, x_[0:-1,2], linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post')
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel(r'$x$')
plt.xlim(0, simulation_time)
plt.ylim(-0.6, 1.1)
plt.legend(loc='upper right')
plt.savefig('../fig/ElGamal/evt_enc_state_feedback/state.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
markers, lines, base = plt.stem([t[i] for i in np.where(et == 1)][0][:], et[et == 1], basefmt=' ')
plt.setp(markers, color=blue, markersize=3.0)
plt.setp(lines, linestyle='-', color=blue, linewidth=1.0)
plt.xlabel('Time (s)')
plt.ylabel('Event')
plt.xlim(0, simulation_time)
plt.ylim(0, 1.05)
plt.yticks([0, 1])
plt.savefig('../fig/ElGamal/evt_enc_state_feedback/trigger.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
markers, lines, base = plt.stem([t[i] for i in np.where(et_ == 1)][0][:], et_[et_ == 1], basefmt=' ')
plt.setp(markers, color=blue, markersize=3.0)
plt.setp(lines, linestyle='-', color=blue, linewidth=1.0)
plt.xlabel('Time (s)')
plt.ylabel('Event')
plt.xlim(0, simulation_time)
plt.ylim(0, 1.05)
plt.yticks([0, 1])
plt.savefig('../fig/ElGamal/evt_enc_state_feedback/enc_trigger.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.show()
