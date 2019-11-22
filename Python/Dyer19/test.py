#! /usr/bin/env python
# coding: utf-8

import Dyer19

cryptosystem = Dyer19.HE1N(128, 1, 32)
cryptosystem.KeyGen()
cryptosystem.Pgen()

gamma = 1e2

x1 = 1.2
x2 = -3.4
x3 = x1 + x2
x4 = x1 * x2

c1 = cryptosystem.Enc(x1, gamma)
c2 = cryptosystem.Enc(x2, gamma)

c3 = cryptosystem.Add(c1, c2)
c4 = cryptosystem.Mult(c1, c2)

y1 = cryptosystem.Dec(c1, gamma)
y2 = cryptosystem.Dec(c2, gamma)
y3 = cryptosystem.Dec(c3, gamma)
y4 = cryptosystem.Dec(c4, gamma ** 2)

print(f"key: p={cryptosystem.p}, kappa={cryptosystem.kappa}, modulus={cryptosystem.modulus}")
print(f"x1={x1}, x2={x2}, x3={x3}, x4={x4}")
print(f"c1={c1}, c2={c2}, c3={c3}, c4={c4}")
print(f"y1={y1}, y2={y2}, y3={y3}, y4={y4}")