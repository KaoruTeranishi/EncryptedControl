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

# plant (discrete time)
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
F = -la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

# cryptosystem
key_length = 256
params, pk, sk = elgamal.keygen(key_length)

# scaling parameter
delta = 0.01

# controller encryption
F_enc = elgamal.enc(params, pk, F, delta)

# state
x = 50 * np.ones([len(t) + 1, n])
x_ = 50 * np.ones([len(t) + 1, n])
x_enc = [[[0, 0] for j in range(n)] for i in range(len(t))]

# input
u = np.zeros([len(t), m])
u_ = np.zeros([len(t), m])
u_enc = [[[[0, 0] for k in range(n)] for j in range(m)] for i in range(len(t))]

# simulation w/o encryption
for k in range(len(t)):
    # controller
    u[k] = F @ x[k]
    # plant update
    x[k+1] = A @ x[k] + B @ u[k]

# simulation w/ encryption
for k in range(len(t)):
    # state encryption
    x_enc[k] = elgamal.enc(params, pk, x_[k], delta)
    # encrypted controller
    u_enc[k] = elgamal.mult(params, F_enc, x_enc[k])
    # input decryption
    u_[k] = elgamal.dec_add(params, sk, u_enc[k], delta ** 2)
    # plant update
    x_[k+1] = A @ x_[k] + B @ u_[k]

# figure
plt.figure()
plt.plot(t, u, linestyle='-', color=gray, linewidth=3.0, label='unencrypted')
plt.plot(t, u_, linestyle='-', color=blue, linewidth=1.0, label='encrypted')
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel(r'$u$')
plt.xlim(0, simulation_time)
plt.legend(loc='lower right')
# plt.savefig('./fig/enc_sf_input.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.figure()
plt.plot(t, x[0:-1,0], linestyle='-', color=gray, linewidth=3.0, label='unencrypted')
plt.plot(t, x[0:-1,1], linestyle='-', color=gray, linewidth=3.0)
plt.plot(t, x_[0:-1,0], linestyle='-', color=blue, linewidth=1.0, label='encrypted')
plt.plot(t, x_[0:-1,1], linestyle='-', color=blue, linewidth=1.0)
plt.plot(t, np.zeros(len(t)), linestyle='--', color=black, linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel(r'$x$')
plt.xlim(0, simulation_time)
plt.legend(loc='upper right')
# plt.savefig('./fig/enc_sf_state.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.tick_params(left=False, labelbottom=False, labelleft=False)
ax1.plot(t, [a[0][0][0] for a in u_enc], linestyle='-', color=blue, linewidth=1.0)
plt.ylabel(r'$c_{1}$')
plt.xlim(0, simulation_time)
ax2 = fig.add_subplot(2, 1, 2)
ax2.tick_params(left=False, labelleft=False)
ax2.plot(t, [a[0][0][1] for a in u_enc], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel('Time (s)')
plt.ylabel(r'$c_{2}$')
plt.xlim(0, simulation_time)
fig.supylabel(r'$\mathsf{Enc}(\Psi_{11})$', fontsize=10)
# plt.savefig('./fig/enc_sf_psi_11.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.tick_params(left=False, labelbottom=False, labelleft=False)
ax1.plot(t, [a[0][1][0] for a in u_enc], linestyle='-', color=blue, linewidth=1.0)
plt.ylabel(r'$c_{1}$')
plt.xlim(0, simulation_time)
ax2 = fig.add_subplot(2, 1, 2)
ax2.tick_params(left=False, labelleft=False)
ax2.plot(t, [a[0][1][1] for a in u_enc], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel('Time (s)')
plt.ylabel(r'$c_{2}$')
plt.xlim(0, simulation_time)
fig.supylabel(r'$\mathsf{Enc}(\Psi_{12})$', fontsize=10)
# plt.savefig('./fig/enc_sf_psi_12.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.tick_params(left=False, labelbottom=False, labelleft=False)
ax1.plot(t, [a[0][0] for a in x_enc], linestyle='-', color=blue, linewidth=1.0)
plt.ylabel(r'$c_{1}$')
plt.xlim(0, simulation_time)
ax2 = fig.add_subplot(2, 1, 2)
ax2.tick_params(left=False, labelleft=False)
ax2.plot(t, [a[0][1] for a in x_enc], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel('Time (s)')
plt.ylabel(r'$c_{2}$')
plt.xlim(0, simulation_time)
fig.supylabel(r'$\mathsf{Enc}(x_{1})$', fontsize=10)
# plt.savefig('./fig/enc_sf_x_1.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.tick_params(left=False, labelbottom=False, labelleft=False)
ax1.plot(t, [a[1][0] for a in x_enc], linestyle='-', color=blue, linewidth=1.0)
plt.ylabel(r'$c_{1}$')
plt.xlim(0, simulation_time)
ax2 = fig.add_subplot(2, 1, 2)
ax2.tick_params(left=False, labelleft=False)
ax2.plot(t, [a[1][1] for a in x_enc], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel('Time (s)')
plt.ylabel(r'$c_{2}$')
plt.xlim(0, simulation_time)
fig.supylabel(r'$\mathsf{Enc}(x_{2})$', fontsize=10)
# plt.savefig('./fig/enc_sf_x_2.eps', bbox_inches='tight', pad_inches=0.05, transparent=True)

plt.show()