#! /usr/bin/env python
# coding: utf-8

import Paillier
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
x_enc = [[0] * n for i in range(len(t) + 1)]

# input
u = np.zeros([1, m, len(t)]).T
u_ = np.zeros([1, m, len(t)]).T
u_enc = [[0] * m for i in range(len(t))]

# cryptosystem
csys = Paillier.Paillier(64)
csys.KeyGen()

# fixed point number parameter
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
print(f"Ecd(F)={F_pt}")

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
plt.savefig("./fig/input.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

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
plt.savefig("./fig/state.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, [a[0] for a in u_enc], linestyle='-', color=blue, linewidth=3.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$\mathsf{Enc}(u)$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/enc_input.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, [a[0] for a in x_enc[0:-1]], linestyle='-', color=blue, linewidth=3.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$\mathsf{Enc}(x_1)$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/enc_state_1.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, [a[1] for a in x_enc[0:-1]], linestyle='-', color=blue, linewidth=3.0)
plt.xlabel("Time (s)")
plt.ylabel(r"$\mathsf{Enc}(x_2)$")
plt.xlim(0, simulation_time)
plt.savefig("./fig/enc_state_2.eps", dbi=300, bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.show()