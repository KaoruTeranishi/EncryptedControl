#! /usr/bin/env python
# coding: utf-8

import ElGamal
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
A = np.array([[1, 0, 0], [0, 0, 1], [-1, -2, -1]])
B = np.array([[1, 0], [0, 1], [0, 0]])
C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
D = np.array([[0, 0], [0, 0], [0, 0]])
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
F = la.inv(B.T @ X @ B + R) * (B.T @ X @ A)
F = -F

# state
x = np.ones([1, n, len(t) + 1]).T
x_ = np.ones([1, n, len(t) + 1]).T
x_bar = np.ones([1, n, len(t) + 1]).T
x_enc = [[(0, 0)] * n for i in range(len(t) + 1)]

# input
u = np.zeros([1, m, len(t)]).T
u_ = np.zeros([1, m, len(t)]).T
u_enc = [[[(0, 0)] * n for i in range(m)] * 1 for i in range(len(t))]

# error
e = np.zeros([1, n, len(t)]).T
e_ = np.zeros([1, n, len(t)]).T

# event trigger
et = np.zeros([1, len(t)]).T
et_ = np.zeros([1, len(t)]).T

# event time
tk = 0

# cryptosystem
csys = ElGamal.ElGamal(20)
csys.KeyGen()

# scaling parameter
d_max = csys.dmaxSearch()

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

# event-triggering parameter
Q_e = np.diag(np.ones(n))
P_e = dlyap((A + B @ F).T, Q_e)
sigma = (la.norm((A + B @ F).T @ P_e @ B @ F, ord=2) + np.sqrt(la.norm((A + B @ F).T @ P_e @ B @ F, ord=2) ** 2 + min(la.eig(Q_e)[0]) * la.norm(F.T @ B.T @ P_e @ B @ F, ord=2))) / min(la.eig(Q_e)[0])

Q_e = np.diag(np.ones(n))
P_e = dlyap((A + B @ F_bar).T, Q_e)
sigma_ = (la.norm((A + B @ F_bar).T @ P_e @ B @ F_bar, ord=2) + np.sqrt(la.norm((A + B @ F_bar).T @ P_e @ B @ F_bar, ord=2) ** 2 + min(la.eig(Q_e)[0]) * la.norm(F_bar.T @ B.T @ P_e @ B @ F_bar, ord=2))) / min(la.eig(Q_e)[0])

# controller encryption
F_enc = [[(0, 0)] * n for i in range(m)]
for i in range(m):
	for j in range(n):
		F_enc[i][j] = csys.Enc(F[i,j], gamma_c)
print(f"Enc(F)={F_enc}")

# simulation w/o encryption
for k in range(len(t)):
	# error
	e[k,:] = x[tk,:] - x[k,:]
	# event triggering
	if k == 0 or la.norm(x[k,:], ord=2) <= sigma * la.norm(e[k,:], ord=2):
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
	if k == 0 or la.norm(x_[k,:], ord=2) <= sigma_ * la.norm(e_[k,:], ord=2):
		# scaling parameter update
		gamma_p[k] = Theta * d_max / la.norm(x_[k], ord=2) + mu_p
		# state encryption
		for i in range(n):
			x_bar[k][i] = csys.Decode(csys.Encode(x_[k,i], gamma_p[k]), gamma_p[k])
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
		tk = k
		et_[k] = 1
	else:
		u_[k,:] = u_[tk,:]
	# plant update
	x_[k+1,:] = A @ x_[k,:] + B @ u_[k,:]

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
plt.plot(t, u[:,1], linestyle='-', color=gray, linewidth=6.0)
plt.plot(t, u_[:,0], linestyle='-', color=blue, linewidth=3.0, label='w/ encryption')
plt.plot(t, u_[:,1], linestyle='-', color=blue, linewidth=3.0)
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=1.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$u$")
plt.xlim(0, simulation_time)
plt.ylim(-3, 0.5)
plt.legend(loc='lower right')
plt.savefig("./fig/evt_input.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, x[0:-1,0], linestyle='-', color=gray, linewidth=6.0, label='w/o encryption')
plt.plot(t, x[0:-1,1], linestyle='-', color=gray, linewidth=6.0)
plt.plot(t, x[0:-1,2], linestyle='-', color=gray, linewidth=6.0)
plt.plot(t, x_[0:-1,0], linestyle='-', color=blue, linewidth=3.0, label='w/ encryption')
plt.plot(t, x_[0:-1,1], linestyle='-', color=blue, linewidth=3.0)
plt.plot(t, x_[0:-1,2], linestyle='-', color=blue, linewidth=3.0)
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=1.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$x$")
plt.xlim(0, simulation_time)
plt.ylim(-0.6, 1.1)
plt.legend(loc='upper right')
plt.savefig("./fig/evt_state.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
markers, lines, base = plt.stem([t[i] for i in np.where(et == 1)][0][:], et[et == 1], basefmt=' ')
plt.setp(markers, color=blue, markersize=3.0)
plt.setp(lines, linestyle='-', color=blue, linewidth=1.0)
plt.xlabel("Time (s)")
plt.ylabel("Event")
plt.xlim(0, simulation_time)
plt.ylim(0, 1.05)
plt.yticks([0, 1])
plt.savefig("./fig/trigger.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
markers, lines, base = plt.stem([t[i] for i in np.where(et_ == 1)][0][:], et_[et_ == 1], basefmt=' ')
plt.setp(markers, color=blue, markersize=3.0)
plt.setp(lines, linestyle='-', color=blue, linewidth=1.0)
plt.xlabel("Time (s)")
plt.ylabel("Event")
plt.xlim(0, simulation_time)
plt.ylim(0, 1.05)
plt.yticks([0, 1])
plt.savefig("./fig/enc_trigger.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.show()