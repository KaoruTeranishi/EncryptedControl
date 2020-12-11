#! /usr/bin/env python

# This is a sample code for encrypted observer-based dynamic feedback with
# dynamic quantizer. The dynamic quantizer guarantees asymptotic stability
# of an encrypted control system.
# The following is the original paper's information:
# K. Teranishi and K. Kogiso, Dynamic quantizer for encrypted observer-based
# control, IEEE Conference on Decision and Control, 2020.

import eclib.elgamal as elgamal
from eclib.colors import *
import eclib.figsetup
import numpy as np
import numpy.linalg as la
from control.matlab import *
import matplotlib.pyplot as plt

# sampling time
Ts = 0.1

# simulation setting
simulation_time = 60
t = np.linspace(0, simulation_time, int(simulation_time / Ts))

# plant (continuous time)
A = np.array([[0.1, -0.2],
              [  0, -0.5]])
B = np.array([[0],
              [1]])
C = np.array([1, 0])
D = np.array([0])

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

# observer
Q = np.identity(n)
R = np.identity(l)
X, _, _ = dare(A.T, C.T, Q, R)
L = -(la.inv(C @ X @ C.T + R) @ (C @ X @ A.T)).T

# state
x = np.ones([len(t) + 1, n])
x_ = np.ones([len(t) + 1, n])

# state estimation
x_hat = np.ones([len(t) + 1, n])
x_hat_ = np.ones([len(t) + 1, n])

# input
u = np.zeros([len(t), m])
u_ = np.zeros([len(t), m])

# output
y = np.zeros([len(t), l])
y_ = np.zeros([len(t), l])

# controller input
xi = np.zeros([len(t), n + l])
xi_ = np.zeros([len(t), n + l])
xi_bar = np.zeros([len(t), n + l])
xi_enc = [[[0, 0] for j in range(n + l)] for i in range(len(t))]

# controller output
psi = np.zeros([len(t), n + m])
psi_ = np.zeros([len(t), n + m])
psi_enc = [[[[0, 0] for k in range(n + l)] for j in range(n + m)] for i in range(len(t))]

# cryptosystem
key_length = 32
params, pk, sk = elgamal.keygen(key_length)
params.p = 6848919887
params.q = 3424459943
params.g = 2
pk = 5527055734
sk = 1076876626

# maximum width of plaintext space
# dmax = elgamal.get_dmax(params)
dmax = 32

# resolution of quantizer for controller parameter
mu_Phi = 0.0001
Q = np.identity(n)
P = dlyap((A + B @ F).T, Q)
Omega = 2 * (-la.norm((A + B @ F).T @ P @ B) + np.sqrt(la.norm((A + B @ F).T @ P @ B) ** 2 + min(la.eig(Q)[0]) * la.norm(B.T @ P @ B))) / (np.sqrt(m * n) * la.norm(B.T @ P @ B))
delta_Phi = Omega / dmax * mu_Phi

# quantized controller parameter used in design of resolution for controller input
A_bar = np.array(elgamal.decode(params, elgamal.encode(params, A, delta_Phi), delta_Phi))
B_bar = np.array(elgamal.decode(params, elgamal.encode(params, B, delta_Phi), delta_Phi))
C_bar = np.array(elgamal.decode(params, elgamal.encode(params, C, delta_Phi), delta_Phi))
F_bar = np.array(elgamal.decode(params, elgamal.encode(params, F, delta_Phi), delta_Phi))
L_bar = np.array(elgamal.decode(params, elgamal.encode(params, L, delta_Phi), delta_Phi))

# resolution of quantizer for controller input
mu_xi = 0.99
G = np.block([[A + B @ F_bar, B @ F_bar], [np.zeros([n, n]), A_bar + L_bar @ C_bar]])
S = np.identity(2 * n)
R = dlyap(G, S)
Theta = 2 * (-la.norm(R @ G) + np.sqrt(la.norm(R @ G) ** 2 + min(la.eig(S)[0]))) / (np.sqrt(l) * la.norm(L_bar) * dmax) - (n / (np.sqrt(l) * la.norm(L_bar)) + np.sqrt(n)) * delta_Phi
Gamma = np.sqrt(m * n / l) * la.norm(F_bar) * delta_Phi / la.norm(L_bar)
delta_xi = [0] * len(t)

# controller encryption
Ac = A + B @ F + L @ C
Bc = -L
Cc = F
Dc = np.zeros([m, l])
Phi = np.block([[Ac, Bc], [Cc, Dc]])
Phi_enc = elgamal.enc(params, pk, Phi, delta_Phi)

# simulation w/o encryption
for k in range(len(t)):
	# measurement
	y[k,:] = C @ x[k,:]
	# signal composition
	xi[k,0:n] = x_hat[k,:]
	xi[k,n:n+l] = y[k,:]
	# controller
	psi[k,:] = Phi @ xi[k,:]
	# signal decomposition
	x_hat[k+1,:] = psi[k,0:n]
	u[k,:] = psi[k,n:n+m]
	# plant update
	x[k+1,:] = A @ x[k,:] + B @ u[k,:]

# simulation w/ encryption
for k in range(len(t)):
	# measurement
	y_[k,:] = C @ x_[k,:]
	# signal composition
	xi_[k,0:n] = x_hat_[k,:]
	xi_[k,n:n+l] = y_[k,:]
	# scaling parameter update
	delta_xi[k] = (Theta * la.norm(C.T @ la.inv(C @ C.T) * y_[k,:]) - Gamma * la.norm(x_hat_[k,:])) * mu_xi
	# controller input encryption
	xi_enc[k] = elgamal.enc(params, pk, xi_[k], delta_xi[k])
	# encrypted controller
	psi_enc[k] = elgamal.mult(params, Phi_enc, xi_enc[k])
	# controller output decryption
	psi_[k] = elgamal.dec_add(params, sk, psi_enc[k], delta_Phi * delta_xi[k])
	# signal decomposition
	x_hat_[k+1,:] = psi_[k,0:n]
	u_[k,:] = psi_[k,n:n+m]
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
print(f'Gamma = {Gamma}')
print(f'delta_Phi = {delta_Phi}')
print(f'delta_xi(0) = {delta_xi[0]}')
print(f'lambda(G) = {la.eig(G)[0]}')
print('========== encrypted controller ==========')
print(f'Enc(Phi) = {Phi_enc}')

# figure
plt.figure()
plt.plot(t, u, linestyle='-', color=gray, linewidth=3.0, drawstyle='steps-post', label='unencrypted')
plt.plot(t, u_, linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post', label='encrypted')
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel(r'$u$')
plt.xlim(0, simulation_time)
plt.ylim(-0.1, 1.1)
plt.legend(loc='upper right')
plt.savefig('../fig/ElGamal/dynqtz_enc_observer/input.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, y, linestyle='-', color=gray, linewidth=3.0, drawstyle='steps-post', label='unencrypted')
plt.plot(t, y_, linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post', label='encrypted')
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel(r'$y$')
plt.xlim(0, simulation_time)
plt.ylim(-0.1, 1.1)
plt.legend(loc='upper right')
plt.savefig('../fig/ElGamal/dynqtz_enc_observer/output.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, [la.norm(a) for a in y - y_], linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post')
plt.xlabel('Time (s)')
plt.ylabel(r'$\|e\|$')
plt.xlim(0, simulation_time)
plt.ylim(0, 0.00125)
plt.savefig('../fig/ElGamal/dynqtz_enc_observer/quantization_error.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, delta_xi, linestyle='-', color=blue, linewidth=1.0, drawstyle='steps-post')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel(r'$\Delta_{\xi}$')
plt.xlim(0, simulation_time)
plt.ylim(1e-9, 1e-3)
plt.yticks([1e-3, 1e-6, 1e-9])
plt.savefig('../fig/ElGamal/dynqtz_enc_observer/resolution.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.show()
