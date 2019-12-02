#! /usr/bin/env python
# coding: utf-8

from Crypto.Util.number import getPrime
from Crypto.Util.number import isPrime
from Crypto.Util.number import getRandomRange
import math

class ElGamal:

	def __init__(self, bitLength):
		self.bitLength = bitLength

	def KeyGen(self):
		self.p = self.getSafePrime(self.bitLength)
		self.q = (self.p - 1) // 2
		self.g = self.getGenerator()
		# self.s = getRandomRange(0, q) # s=0 is unsecure
		self.s = getRandomRange(1, self.q)
		self.h = self.modPow(self.g, self.s, self.p)

	def Encode(self, x, gamma):
		m = math.floor(x * gamma + 0.5)
		firstDecimalPlace = (x * gamma * 10) % 10

		if m < 0:
			if m < -(self.q + 1):
				print("Underflow.")
				return
			else:
				m += self.p
		elif m >= self.q:
			print("Overflow.")
			return
		
		if self.isElement(m):
			return m
		else:
			if firstDecimalPlace == 0 or firstDecimalPlace >= 5:
				for i in range(self.q):
					if self.isElement(m - i):
						return m - i
					elif self.isElement(m + i):
						return m + i
			else:
				for i in range(self.q):
					if self.isElement(m + i):
						return m + i
					elif self.isElement(m - i):
						return m - i

		print("Failed to encode.")

	def Decode(self, m, gamma):
		return (m - self.p) / gamma if m >= self.q else m / gamma

	def Encrypt(self, m):
		r = getRandomRange(1, self.q)
		return self.modPow(self.g, r, self.p), (m * self.modPow(self.h, r, self.p)) % self.p

	def Decrypt(self, c):
		return (c[1] * self.modInv(self.modPow(c[0], self.s, self.p), self.p)) % self.p

	def Enc(self, x, gamma):
		return self.Encrypt(self.Encode(x, gamma))

	def Dec(self, c, gamma):
		return self.Decode(self.Decrypt(c), gamma)

	def Mult(self, c1, c2):
		return (c1[0] * c2[0]) % self.p, (c1[1] * c2[1]) % self.p

	def modPow(self, a, b, m):
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
	
	def getSafePrime(self, bitLength):
		q = getPrime(bitLength)
		while not isPrime(2 * q + 1):
			q = getPrime(bitLength)

		return 2 * q + 1

	def getGenerator(self):
		g = 2
		while not self.isGenerator(g):
			g += 1

		if g >= self.p:
			print("Failed to get generator.")
			return

		return g

	def isGenerator(self, g):
		return True if self.modPow(g, self.q, self.p) == 1 else False

	def isElement(self, m):
		return True if self.modPow(m, self.q, self.p) == 1 else False
