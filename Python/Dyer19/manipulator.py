#! /usr/bin/env python
# coding: utf-8

import math
import numpy as np
from numpy import linalg as la

class oneLinkManipulator:
	def __init__(self, m, l, theta_init, theta_dot_init, dt):
		self.m = m
		self.l = l
		self.theta = theta_init
		self.theta_dot = theta_dot_init
		self.g = 9.81
		self.dt = dt

	def update(self, tau):
		A = np.array([[0, 1], [0, 0]])
		B = np.array([[0], [1]])
		x = np.array([[self.theta], [self.theta_dot]])
		u = 1.0 / (self.m * pow(self.l, 2)) * tau - self.g / self.l * math.sin(self.theta)

		# Runge-Kutta (4)
		k_1 = (A @ x + B * u) * self.dt
		k_2 = (A @ (x + k_1 / 2) + B * u) * self.dt
		k_3 = (A @ (x + k_2 / 2) + B * u) * self.dt
		k_4 = (A @ (x + k_3) + B * u) * self.dt
		x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
		self.theta = x[0][0]
		self.theta_dot = x[1][0]

	def getPosition(self):
		return self.l * math.cos(self.theta), self.l * math.sin(self.theta)

class twoLinkManipulator:
	def __init__(self, m, l, theta_init, theta_dot_init, dt):
		self.m = np.array(m)
		self.l = np.array(l)
		self.theta = np.array(theta_init)
		self.theta_dot = np.array(theta_dot_init)
		self.g = 9.81
		self.dt = dt

	def update(self, tau):
		M_11 = (self.m[0] + self.m[1]) * pow(self.l[0], 2) + self.m[1] * pow(self.l[1], 2) + 2 * self.m[1] * self.l[0] * self.l[1] * math.cos(self.theta[1])
		M_12 = self.m[1] * pow(self.l[1], 2) + self.m[1] * self.l[0] * self.l[1] * math.cos(self.theta[1])
		M_21 = self.m[1] * pow(self.l[1], 2) + self.m[1] * self.l[0] * self.l[1] * math.cos(self.theta[1])
		M_22 = self.m[1] * pow(self.l[1], 2)
		M = np.array([[M_11, M_12], [M_21, M_22]])

		V_1 = -self.m[1] * self.l[0] * self.l[1] * (2 * self.theta_dot[0] * self.theta_dot[1] + pow(self.theta_dot[1], 2)) * math.sin(self.theta[1])
		V_2 = self.m[1] * self.l[0] * self.l[1] * pow(self.theta_dot[0], 2) * math.sin(self.theta[1])
		V = np.array([V_1, V_2])

		G_1 = (self.m[0] + self.m[1]) * self.g * self.l[0] * math.cos(self.theta[0]) + self.m[1] * self.g * self.l[1] * math.cos(self.theta[0] + self.theta[1])
		G_2 = self.m[1] * self.g * self.l[1] * math.cos(self.theta[0] + self.theta[1])
		G = np.array([G_1, G_2])

		# Runge-Kutta (4)
		A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
		B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
		x = np.array([[self.theta[0]], [self.theta[1]], [self.theta_dot[0]], [self.theta_dot[1]]])
		u = la.inv(M) @ (tau - V - G)
		u = u.reshape(2, 1)
		k_1 = (A @ x + B @ u) * self.dt
		k_2 = (A @ (x + k_1 / 2) + B @ u) * self.dt
		k_3 = (A @ (x + k_2 / 2) + B @ u) * self.dt
		k_4 = (A @ (x + k_3) + B @ u) * self.dt
		x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
		self.theta[0] = x[0]
		self.theta[1] = x[1]
		self.theta_dot[0] = x[2]
		self.theta_dot[1] = x[3]

	def getPosition(self):
		return self.l[0] * math.cos(self.theta[0]) + self.l[1] * math.cos(self.theta[0] + self.theta[1]), self.l[0] * math.sin(self.theta[0]) + self.l[1] * math.sin(self.theta[0] + self.theta[1])

class RPManipulator:
	def __init__(self, m, l, I, q_init, q_dot_init, dt):
		self.m = np.array(m)
		self.l = np.array(l)
		self.I = np.array(I)
		self.q = np.array(q_init)
		self.q_dot = np.array(q_dot_init)
		self.g = 9.81
		self.dt = dt

	def update(self, tau):
		M_11 = self.m[0] * pow(self.l[0], 2) + self.I[0] + self.m[1] * pow(self.q[1], 2) + self.I[1]
		M_12 = 0
		M_21 = 0
		M_22 = self.m[1]
		M = np.array([[M_11, M_12], [M_21, M_22]])

		V_1 = 2 * self.m[1] * self.q[1] * self.q_dot[1] * self.q_dot[0]
		V_2 = -self.m[1] * self.q[1] * pow(self.q[0], 2)
		V = np.array([V_1, V_2])

		G_1 = (self.m[0] * self.l[0] + self.m[1] * self.q[1]) * self.g * math.cos(self.q[0])
		G_2 = self.m[1] * self.g * math.sin(self.q[0])
		G = np.array([G_1, G_2])

		# Runge-Kutta (4)
		A = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
		B = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])
		x = np.array([[self.q[0]], [self.q_dot[0]], [self.q[1]], [self.q_dot[1]]])
		u = la.inv(M) @ (tau - V - G)
		u = u.reshape(2, 1)
		k_1 = (A @ x + B @ u) * self.dt
		k_2 = (A @ (x + k_1 / 2) + B @ u) * self.dt
		k_3 = (A @ (x + k_2 / 2) + B @ u) * self.dt
		k_4 = (A @ (x + k_3) + B @ u) * self.dt
		x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
		self.q[0] = x[0]
		self.q_dot[0] = x[1]
		self.q[1] = x[2]
		self.q_dot[1] = x[3]

	def getPosition(self):
		return (self.q[1] + self.l[1]) * math.cos(self.q[0]), (self.q[1] + self.l[1]) * math.sin(self.q[0])