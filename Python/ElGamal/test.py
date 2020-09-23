#! /usr/bin/env python
# coding: utf-8

import ElGamal

key_length = 64
csys = ElGamal.ElGamal(key_length)
csys.KeyGen()

gamma = 1e2

x1 = 1.23
x2 = -4.56
x3 = x1 * x2

c1 = csys.Enc(x1, gamma)
c2 = csys.Enc(x2, gamma)

c3 = csys.Mult(c1, c2)

y1 = csys.Dec(c1, gamma)
y2 = csys.Dec(c2, gamma)
y3 = csys.Dec(c3, gamma ** 2)

print(f"key length = {key_length}")
print("public key:")
print(f"p = {csys.p}")
print(f"q = {csys.q}")
print(f"g = {csys.g}")
print(f"h = {csys.h}")
print("secret key:")
print(f"s = {csys.s}")
print("computation test:")
print(f"x1 = {x1}, x2 = {x2}, x3 = {x3}")
print(f"c1 = {c1}, c2 = {c2}, c3 = {c3}")
print(f"y1 = {y1}, y2 = {y2}, y3 = {y3}")