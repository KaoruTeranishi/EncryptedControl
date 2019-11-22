#! /usr/bin/env python
# coding: utf-8

import ElGamal

cryptosystem = ElGamal.ElGamal(64)
cryptosystem.KeyGen()

gamma = 1e2

x1 = 1.23
x2 = -4.56
x3 = x1 * x2

c1 = cryptosystem.Enc(x1, gamma)
c2 = cryptosystem.Enc(x2, gamma)

c3 = cryptosystem.Mult(c1, c2)

y1 = cryptosystem.Dec(c1, gamma)
y2 = cryptosystem.Dec(c2, gamma)
y3 = cryptosystem.Dec(c3, gamma ** 2)

print(f"pk: p={cryptosystem.p}, q={cryptosystem.q}, g={cryptosystem.g}, h={cryptosystem.h}")
print(f"sk: s={cryptosystem.s}")
print(f"x1={x1}, x2={x2}, x3={x3}")
print(f"c1={c1}, c2={c2}, c3={c3}")
print(f"y1={y1}, y2={y2}, y3={y3}")