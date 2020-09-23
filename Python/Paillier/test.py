#! /usr/bin/env python
# coding: utf-8

import Paillier

key_length = 64
csys = Paillier.Paillier(key_length)
csys.KeyGen()

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

c1 = csys.Enc(x1, a, b, a1, b1)
c2 = csys.Enc(x2, a, b, a2, b2)

c3 = csys.Add(c1, c2)
c4 = csys.Mult(c1, csys.Encode(x2, a, b, a2, b2))

y1 = csys.Dec_(c1, a, b)
y2 = csys.Dec_(c2, a, b)
y3 = csys.Dec_(c3, a, b)
y4 = csys.Dec(c4, a, b)

print(f"key length = {key_length}")
print("public key:")
print(f"n = {csys.n}")
print(f"g = {csys.g}")
print("secret key:")
print(f"p = {csys.p}")
print(f"q = {csys.q}")
print(f"lambda = {csys.lmd}")
print(f"mu = {csys.mu}")
print("computation test:")
print(f"x1 = {x1}, x2 = {x2}, x3 = {x3}, x4 = {x4}")
print(f"c1 = {c1}, c2 = {c2}, c3 = {c3}, c4 = {c4}")
print(f"y1 = {y1}, y2 = {y2}, y3 = {y3}, y4 = {y4}")