#! /usr/bin/env python
# coding: utf-8

import Paillier

cryptosystem = Paillier.Paillier(64)
cryptosystem.KeyGen()

x1 = -1.23
x2 = -4.56
x3 = x1 + x2
x4 = x1 * x2

n = 1
a1 = 9
b1 = 7
a2 = 9
b2 = 8
a = a1 + a2 + n
b = b1 + b2

c1 = cryptosystem.Enc(x1, a, b, a1, b1)
c2 = cryptosystem.Enc(x2, a, b, a2, b2)

c3 = cryptosystem.Add(c1, c2)
c4 = cryptosystem.Mult(c1, cryptosystem.Encode(x2, a, b, a2, b2))

y1 = cryptosystem.Dec_(c1, a, b)
y2 = cryptosystem.Dec_(c2, a, b)
y3 = cryptosystem.Dec_(c3, a, b)
y4 = cryptosystem.Dec(c4, a, b)

print(f"p={cryptosystem.p}, q={cryptosystem.q}, n={cryptosystem.n}, n^2={cryptosystem.n_square}, lambda={cryptosystem.lmd}, g={cryptosystem.g}, mu={cryptosystem.mu}")
print(f"x1={x1}, x2={x2}, x3={x3}, x4={x4}")
print(f"c1={c1}, c2={c2}, c3={c3}, c4={c4}")
print(f"y1={y1}, y2={y2}, y3={y3}, y4={y4}")