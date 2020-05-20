#! /usr/bin/env python
# coding: utf-8

import ElGamal
from Crypto.Util.number import getRandomRange
import numpy as np
import numpy.linalg as la
from control.matlab import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# setting
Ts = 10e-3
simulation_time = 10
t = np.linspace(0, simulation_time - Ts, simulation_time / Ts)

# plant
A = np.array([[1, -1], [0, 2]])
B = np.array([[0], [1]])
C = np.array([[1, 0], [0, 1]])
D = np.array([[0], [0]])
sys = c2d(ss(A, B, C, D), Ts)
A = sys.A
B = sys.B
C = sys.C
D = sys.D

# dimension
n = A.shape[1]
m = B.shape[1]

# controller
Q = np.diag(np.ones(n))
R = np.diag(np.ones(m))
X, _, _ = dare(A, B, Q, R)
F = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)
F = -F

# state
x = np.ones([1, n, len(t) + 1]).T
x_ = np.ones([1, n, len(t) + 1]).T
x_enc = [[(0, 0)] * n for i in range(len(t) + 1)]

# input
u = np.zeros([1, m, len(t)]).T
u_ = np.zeros([1, m, len(t)]).T
u_enc = [[[(0, 0)] * n for i in range(m)] * 1 for i in range(len(t))]

# cryptosystem
csys = ElGamal.DynElGamal(20)
csys.KeyGen()

# scaling parameter
d_max = csys.dmaxSearch()
print(f"dmax={d_max}")

mu_c = 0.01
Q_c = np.diag(np.ones(n))
P_c = dlyap((A + B @ F).T, Q_c)
Omega = 2 * (-la.norm((A + B @ F).T @ P_c @ B, ord=2) + np.sqrt(la.norm((A + B @ F).T @ P_c @ B, ord=2) ** 2 + min(la.eig(Q_c)[0]) * la.norm(B.T @ P_c @ B, ord=2))) / (np.sqrt(m * n) * la.norm(B.T @ P_c @ B, ord=2))
gamma_c = d_max / Omega + mu_c

F_bar = F
for i in range(m):
	for j in range(n):
		F_bar[i,j] = csys.Decode(csys.Encode(F[i,j], gamma_c), gamma_c)

mu_p = 0.01
Q_p = np.diag(np.ones(n))
P_p = dlyap((A + B @ F_bar).T, Q_c)
Theta = np.sqrt(n) * (la.norm((A + B @ F_bar).T @ P_p @ B @ F_bar, ord=2) + np.sqrt(la.norm((A + B @ F_bar).T @ P_p @ B @ F_bar, ord=2) ** 2 + min(la.eig(Q_p)[0]) * la.norm(F_bar.T @ B.T @ P_p @ B @ F_bar, ord=2))) / (2 * min(la.eig(Q_p)[0]))
gamma_p = [0] * len(t)

# list for plot
pk = [csys.h] * (len(t) + 1)
sk = [csys.s] * (len(t) + 1)

# controller encryption
F_enc = [[(0, 0)] * n for i in range(m)]
for i in range(m):
	for j in range(n):
		F_enc[i][j] = csys.Enc(F[i,j], gamma_c)
print(f"Enc(F)={F_enc[0]}")

# simulation w/o encryption
for k in range(len(t)):
	# controller
	u[k,:] = F @ x[k,:]
	# plant update
	x[k+1,:] = A @ x[k,:] + B @ u[k,:]

# simulation w/ encryption
for k in range(len(t)):
	# scaling parameter update
	gamma_p[k] = Theta * d_max / la.norm(x_[k], ord=2) + mu_p
	# state encryption
	for i in range(n):
		x_enc[k][i] = csys.Enc(x_[k,i], gamma_p[k])
	# encrypted controller
	for i in range(m):
		for j in range(n):
			u_enc[k][i][j] = csys.Mult(F_enc[i][j], x_enc[k][j])
	# input decryption
	for i in range(m):
		u_[k,i] = csys.Dec(u_enc[k][i][0], gamma_c * gamma_p[k])
		for j in range(1, n):
			u_[k,i] += csys.Dec(u_enc[k][i][j], gamma_c * gamma_p[k])
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

# figure property
# color
red = (1, 75 / 255, 0)
yellow = (1, 241 / 255, 0)
green = (3 / 255, 175 / 255, 122 / 255)
blue = (0 , 90 / 255, 1)
sky_blue = (77 / 255, 196 / 255, 1)
pink = (1, 128 / 255, 130 / 255)
orange = (246 / 255, 170 / 255, 0)
purple = (153 / 255, 0, 153 / 255)
brown = (128 / 255, 64 / 255, 0)
black = (0, 0, 0)
gray = (132 / 255, 145 / 255, 158 / 255)
light_gray = (200 / 255, 200 / 255, 203 / 255)
white = (1, 1, 1)
# size
plt.rcParams['figure.figsize'] = (1.62 * 3, 1 * 3)
# font
del fm.weight_dict['roman']
matplotlib.font_manager._rebuild()
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
# axis
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
# legend
plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.edgecolor'] = '#000000'
plt.rcParams['legend.handlelength'] = 1.0
# grid
plt.rcParams['axes.grid'] = False

# figure
plt.figure()
plt.plot(t, u[:,0], linestyle='-', color=gray, linewidth=6.0, label='w/o encryption')
plt.plot(t, u_[:,0], linestyle='-', color=blue, linewidth=3.0, label='w/ encryption')
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=1.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$u$")
plt.xlim(0, simulation_time)
plt.ylim(-3.3, 0.5)
plt.legend(loc='lower right')
plt.savefig("./fig/dynqtz_dynenc_input.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, x[0:-1,0], linestyle='-', color=gray, linewidth=6.0, label='w/o encryption')
plt.plot(t, x[0:-1,1], linestyle='-', color=gray, linewidth=6.0)
plt.plot(t, x_[0:-1,0], linestyle='-', color=blue, linewidth=3.0, label='w/ encryption')
plt.plot(t, x_[0:-1,1], linestyle='-', color=blue, linewidth=3.0)
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=1.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$x$")
plt.xlim(0, simulation_time)
plt.ylim(-0.05, 1.45)
plt.legend(loc='upper right')
plt.savefig("./fig/dynqtz_dynenc_state.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, [a[0][0] for a in u_enc], linestyle='-', color=blue, linewidth=3.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$\mathsf{Enc}(u)_1$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/dynqtz_dynenc_enc_input_1.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, [a[0][1] for a in u_enc], linestyle='-', color=blue, linewidth=3.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$\mathsf{Enc}(u)_2$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/dynqtz_dynenc_enc_input_2.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, [a[0][0] for a in x_enc[0:-1]], linestyle='-', color=blue, linewidth=3.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$\mathsf{Enc}(x_1)_1$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/dynqtz_dynenc_enc_state_11.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, [a[0][1] for a in x_enc[0:-1]], linestyle='-', color=blue, linewidth=3.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$\mathsf{Enc}(x_1)_2$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/dynqtz_dynenc_enc_state_12.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, [a[1][0] for a in x_enc[0:-1]], linestyle='-', color=blue, linewidth=3.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$\mathsf{Enc}(x_2)_1$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/dynqtz_dynenc_enc_state_21.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, [a[1][1] for a in x_enc[0:-1]], linestyle='-', color=blue, linewidth=3.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$\mathsf{Enc}(x_2)_2$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/dynqtz_dynenc_enc_state_22.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, pk[0:-1], linestyle='-', color=blue, linewidth=3.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$\mathsf{pk}$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/dynqtz_dynenc_pk.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, sk[0:-1], linestyle='-', color=blue, linewidth=3.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$\mathsf{sk}$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/dynqtz_dynenc_sk.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, gamma_p, linestyle='-', color=blue, linewidth=3.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$\gamma_p$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/dynqtz_dynenc_gamma_p.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.show()