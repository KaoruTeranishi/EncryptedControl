#! /usr/bin/env python
# coding: utf-8

from Crypto.Util.number import getPrime
from Crypto.Util.number import getRandomRange
import math

class HE1:

	def __init__(self, bits, entropy):
		self.bits = bits
		self.rho = entropy

	def KeyGen(self):
		self.p = getPrime(self.bits)

	def Pgen(self):
		eta = self.bits ** 2 // self.rho - self.bits
		self.q = getPrime(eta)
		self.modulus = self.p * self.q

	def Encode(self, x, gamma):
		return math.floor(x * gamma + 0.5)

	def Decode(self, m, gamma):
		return m / gamma

	def Encrypt(self, m):
		r = getRandomRange(1, self.q)
		return (m + r * self.p) % self.modulus

	def Decrypt(self, c):
		return self.minimalResidue(c, self.p)

	def Enc(self, x, gamma):
		return self.Encrypt(self.Encode(x, gamma))

	def Dec(self, c, gamma):
		return self.Decode(self.Decrypt(c), gamma)

	def Add(self, c1, c2):
		return (c1 + c2) % self.modulus

	def Mult(self, c1, c2):
		return (c1 * c2) % self.modulus

	def minimalResidue(self, x, m):
		a = x % m
		b = a - m
		if a >= abs(b):
			return b
		else:
			return a

	def modInv(self, a, m):
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

class HE1N(HE1):

	def __init__(self, bits, prev_entropy, new_entropy):
		self.bits = bits
		self.rho = prev_entropy
		self.rho_prime = new_entropy

	def KeyGen(self):
		nu = self.rho_prime - self.rho
		self.p = getPrime(self.bits)
		self.kappa = getPrime(nu)

	def Pgen(self):
		eta = self.bits ** 2 // self.rho_prime - self.bits
		self.q = getPrime(eta)
		self.modulus = self.p * self.q

	def Encrypt(self, m):
		r = getRandomRange(1, self.q)
		s = getRandomRange(0, self.kappa)
		return (m + s * self.kappa + r * self.p) % self.modulus

	def Decrypt(self, c):
		return self.minimalResidue(self.minimalResidue(c, self.p), self.kappa)