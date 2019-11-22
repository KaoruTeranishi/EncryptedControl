#! /usr/bin/env python
# coding: utf-8

from Crypto.Util.number import getPrime
from Crypto.Util.number import getRandomRange
import math

class Paillier:

	def __init__(self, bitLength):
		self.bitLength = bitLength

	def KeyGen(self):
		self.p = getPrime(self.bitLength)
		self.q = getPrime(self.bitLength)
		while math.gcd(self.p * self.q, (self.p - 1) * (self.q - 1)) != 1 or self.p == self.q:
			self.p = getPrime(self.bitLength)
			self.q = getPrime(self.bitLength)

		self.n = self.p * self.q
		self.n_square = pow(self.n, 2)

		self.lmd = self.lcm(self.p - 1, self.q - 1)

		k = getRandomRange(0, self.n)
		while math.gcd(k, self.n) != 1:
			k = getRandomRange(0, self.n)
		# k = 1 # original paper version
		self.g = (k * self.n + 1) % self.n_square

		self.mu = self.ModInv(self.L(self.ModPow(self.g, self.lmd, self.n_square)), self.n)

	def Encode(self, x, a, b, a_, b_):
		if x < -pow(2, a_):
			print("Underflow.")
			return
		elif x > pow(2, a_) - pow(2, -b_):
			print("Overflow.")
			return
		else:
			q = math.floor(x * pow(2, b_) + 0.5) / pow(2, b_) # encoding to fixed point number
			return (pow(2, b) * q) % pow(2, a + 2 * b)

	def Decode(self, m, a, b):
		return self.minimalResidue(m, pow(2, a + 2 * b)) / pow(2, 2 * b)
		# return (m % pow(2, a + 2 * b)) / pow(2, 2 * b) # original paper version

	def Decode_(self, m, a, b):
		return self.minimalResidue(m, pow(2, a + 2 * b)) / pow(2, b)
		# return (m % pow(2, a + 2 * b)) / pow(2, b) # original paper version

	def Encrypt(self, m):
		r = getRandomRange(0, self.n)
		while math.gcd(r, self.n) != 1:
			r = getRandomRange(0, self.n)

		return (self.ModPow(self.g, m, self.n_square) * self.ModPow(r, self.n, self.n_square)) % self.n_square

	def Decrypt(self, c):
		return (self.L(self.ModPow(c, self.lmd, self.n_square)) * self.mu) % self.n

	def Enc(self, x, a, b, a_, b_):
		return self.Encrypt(self.Encode(x, a, b, a_, b_))

	def Dec(self, c, a, b):
		return self.Decode(self.Decrypt(c), a, b)

	def Dec_(self, c, a, b):
		return self.Decode_(self.Decrypt(c), a, b)

	def Add(self, c1, c2):
		return (c1 * c2) % self.n_square

	def Mult(self, c, m):
		return self.ModPow(c, m, self.n_square)

	def lcm(self, a, b):
		return (a * b) // math.gcd(a, b)

	def L(self, x):
		return (x - 1) // self.n

	def ModPow(self, a, b, m):
		# Bruce Schneier algorithm
		if a < 0:
			a += m

		c = 1
		while b >= 1:
			if b % 2 == 1:
				c = (a * c) % m
			a = pow(a, 2) % m
			b = b // 2

		return c

	def ModInv(self, a, m):
		# extended Euclidean algorithm
		b = m
		u = 1
		v = 0

		while b != 0:
			t = a // b
			a = a - t * b
			u = u - t * v
			a, b = b, a
			u, v = v, u

		return u + m if u < 0 else u

	def minimalResidue(self, x, m):
		a = x % m
		b = a - m
		if a >= abs(b):
			return b
		else:
			return a