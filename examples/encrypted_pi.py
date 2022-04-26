from eclib.elgamal import *
from eclib.colors import *
import eclib.figsetup
import numpy as np
import numpy.linalg as la
from control.matlab import *
import matplotlib.pyplot as plt

# sampling time
Ts = 0.1

# simulation setting
simulation_time = 10
t = np.linspace(0, simulation_time - Ts, int(simulation_time / Ts))

# plant (continuous time)
A = np.array([[-10, -2.5],
              [  2,  0  ]])
B = np.array([[0.5],
              [0  ]])
C = np.array([0, 1])
D = np.array(0)

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
Kp = 15.34
Ki = 15.62

Phi = np.array([[ 1, Ts, -Ts],
                [Ki, Kp, -Kp]])

# cryptosystem
key_length = 256
params, pk, sk = keygen(key_length)

# scaling parameter
delta = 0.0001

# controller encryption
Phi_enc = enc(params, pk, Phi, delta)

# state
x = np.zeros([len(t) + 1, n])
x_ = np.zeros([len(t) + 1, n])

# input
u = np.zeros([len(t), m])
u_ = np.zeros([len(t), m])

# output
y = np.zeros([len(t), l])
y_ = np.zeros([len(t), l])

# reference
r = np.zeros([len(t), l])
r_ = np.zeros([len(t), l])

# controller state
w = np.zeros([len(t) + 1, l])
w_ = np.zeros([len(t) + 1, l])

# controller input
xi = np.zeros([len(t), 3 * l])
xi_ = np.zeros([len(t), 3 * l])
xi_enc = [[[0, 0] for j in range(3 * l)] for i in range(len(t))]

# controller output
psi = np.zeros([len(t), l + m])
psi_ = np.zeros([len(t), l + m])
psi_enc = [[[[0, 0] for k in range(3 * l)] for j in range(l + m)] for i in range(len(t))]

# unencrypted control
for k in range(len(t)):
    # reference
    r[k] = 1
    # sensor measurement
    y[k] = C @ x[k]
    # controller input
    xi[k,0:l] = w[k]
    xi[k,l:2*l] = r[k]
    xi[k,2*l:3*l] = y[k]
    # controller computation
    psi[k] = Phi @ xi[k]
    # controller output
    w[k+1] = psi[k,0:l]
    u[k] = psi[k,l:l+m]
    # plant update
    x[k+1] = A @ x[k] + B @ u[k]

# encrypted control
for k in range(len(t)):
    # reference
    r_[k] = 1
    # sensor measurement
    y_[k] = C @ x_[k]
    # controller input
    xi_[k,0:l] = w_[k]
    xi_[k,l:2*l] = r_[k]
    xi_[k,2*l:3*l] = y_[k]
    xi_enc[k] = enc(params, pk, xi_[k], delta)
    # encrypted controller computation
    psi_enc[k] = mult(params, Phi_enc, xi_enc[k])
    # controller output
    psi_[k] = dec_add(params, sk, psi_enc[k], delta ** 2)
    w_[k+1] = psi_[k,0:l]
    u_[k] = psi_[k,l:l+m]
    # plant update
    x_[k+1] = A @ x_[k] + B @ u_[k]

plt.figure()
plt.plot(t, u, linestyle='-', color=gray, linewidth=3.0, label='unencrypted')
plt.plot(t, u_, linestyle='-', color=blue, linewidth=1.0, label='encrypted')
plt.xlabel('Time (s)')
plt.ylabel(r'$u$')
plt.xlim(0, simulation_time)
plt.ylim(0, 17)
plt.legend(loc='upper right')

plt.figure()
plt.plot(t, y, linestyle='-', color=gray, linewidth=3.0, label='unencrypted')
plt.plot(t, y_, linestyle='-', color=blue, linewidth=1.0, label='encrypted')
plt.plot(t, r, linestyle='--', color=black, linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel(r'$y$')
plt.xlim(0, simulation_time)
plt.ylim(0, 1.2)
plt.legend(loc='lower right')

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.tick_params(left=False, labelbottom=False, labelleft=False)
ax1.plot(t, [a[0][0][0] for a in psi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.ylabel(r'$c_{1}$')
plt.xlim(0, simulation_time)
ax2 = fig.add_subplot(2, 1, 2)
ax2.tick_params(left=False, labelleft=False)
ax2.plot(t, [a[0][0][1] for a in psi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel('Time (s)')
plt.ylabel(r'$c_{2}$')
plt.xlim(0, simulation_time)
fig.supylabel(r'$\mathsf{Enc}(\Psi_{11})$', fontsize=10)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.tick_params(left=False, labelbottom=False, labelleft=False)
ax1.plot(t, [a[0][1][0] for a in psi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.ylabel(r'$c_{1}$')
plt.xlim(0, simulation_time)
ax2 = fig.add_subplot(2, 1, 2)
ax2.tick_params(left=False, labelleft=False)
ax2.plot(t, [a[0][1][1] for a in psi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel('Time (s)')
plt.ylabel(r'$c_{2}$')
plt.xlim(0, simulation_time)
fig.supylabel(r'$\mathsf{Enc}(\Psi_{12})$', fontsize=10)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.tick_params(left=False, labelbottom=False, labelleft=False)
ax1.plot(t, [a[0][2][0] for a in psi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.ylabel(r'$c_{1}$')
plt.xlim(0, simulation_time)
ax2 = fig.add_subplot(2, 1, 2)
ax2.tick_params(left=False, labelleft=False)
ax2.plot(t, [a[0][2][1] for a in psi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel('Time (s)')
plt.ylabel(r'$c_{2}$')
plt.xlim(0, simulation_time)
fig.supylabel(r'$\mathsf{Enc}(\Psi_{13})$', fontsize=10)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.tick_params(left=False, labelbottom=False, labelleft=False)
ax1.plot(t, [a[1][0][0] for a in psi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.ylabel(r'$c_{1}$')
plt.xlim(0, simulation_time)
ax2 = fig.add_subplot(2, 1, 2)
ax2.tick_params(left=False, labelleft=False)
ax2.plot(t, [a[1][0][1] for a in psi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel('Time (s)')
plt.ylabel(r'$c_{2}$')
plt.xlim(0, simulation_time)
fig.supylabel(r'$\mathsf{Enc}(\Psi_{21})$', fontsize=10)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.tick_params(left=False, labelbottom=False, labelleft=False)
ax1.plot(t, [a[1][1][0] for a in psi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.ylabel(r'$c_{1}$')
plt.xlim(0, simulation_time)
ax2 = fig.add_subplot(2, 1, 2)
ax2.tick_params(left=False, labelleft=False)
ax2.plot(t, [a[1][1][1] for a in psi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel('Time (s)')
plt.ylabel(r'$c_{2}$')
plt.xlim(0, simulation_time)
fig.supylabel(r'$\mathsf{Enc}(\Psi_{22})$', fontsize=10)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.tick_params(left=False, labelbottom=False, labelleft=False)
ax1.plot(t, [a[1][2][0] for a in psi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.ylabel(r'$c_{1}$')
plt.xlim(0, simulation_time)
ax2 = fig.add_subplot(2, 1, 2)
ax2.tick_params(left=False, labelleft=False)
ax2.plot(t, [a[1][2][1] for a in psi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel('Time (s)')
plt.ylabel(r'$c_{2}$')
plt.xlim(0, simulation_time)
fig.supylabel(r'$\mathsf{Enc}(\Psi_{23})$', fontsize=10)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.tick_params(left=False, labelbottom=False, labelleft=False)
ax1.plot(t, [a[0][0] for a in xi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.ylabel(r'$c_{1}$')
plt.xlim(0, simulation_time)
ax2 = fig.add_subplot(2, 1, 2)
ax2.tick_params(left=False, labelleft=False)
ax2.plot(t, [a[0][1] for a in xi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel('Time (s)')
plt.ylabel(r'$c_{2}$')
plt.xlim(0, simulation_time)
fig.supylabel(r'$\mathsf{Enc}(\xi_{1})$', fontsize=10)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.tick_params(left=False, labelbottom=False, labelleft=False)
ax1.plot(t, [a[1][0] for a in xi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.ylabel(r'$c_{1}$')
plt.xlim(0, simulation_time)
ax2 = fig.add_subplot(2, 1, 2)
ax2.tick_params(left=False, labelleft=False)
ax2.plot(t, [a[1][1] for a in xi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel('Time (s)')
plt.ylabel(r'$c_{2}$')
plt.xlim(0, simulation_time)
fig.supylabel(r'$\mathsf{Enc}(\xi_{2})$', fontsize=10)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.tick_params(left=False, labelbottom=False, labelleft=False)
ax1.plot(t, [a[2][0] for a in xi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.ylabel(r'$c_{1}$')
plt.xlim(0, simulation_time)
ax2 = fig.add_subplot(2, 1, 2)
ax2.tick_params(left=False, labelleft=False)
ax2.plot(t, [a[2][1] for a in xi_enc], linestyle='-', color=blue, linewidth=1.0)
plt.xlabel('Time (s)')
plt.ylabel(r'$c_{2}$')
plt.xlim(0, simulation_time)
fig.supylabel(r'$\mathsf{Enc}(\xi_{3})$', fontsize=10)

plt.show()