#! /usr/bin/env python
# coding: utf-8

import Dyer19

bits = 128
rho = 1
rho_prime = 32
csys = Dyer19.HE1N(bits, rho, rho_prime)
csys.KeyGen()
csys.Pgen()

gamma = 1e2

x1 = 1.2
x2 = -3.4
x3 = x1 + x2
x4 = x1 * x2

c1 = csys.Enc(x1, gamma)
c2 = csys.Enc(x2, gamma)

c3 = csys.Add(c1, c2)
c4 = csys.Mult(c1, c2)

y1 = csys.Dec(c1, gamma)
y2 = csys.Dec(c2, gamma)
y3 = csys.Dec(c3, gamma)
y4 = csys.Dec(c4, gamma ** 2)

print(f"bits = {csys.bits}")
print(f"rho = {csys.rho}")
print(f"rho' = {csys.rho_prime}")
print("public parameter:")
print(f"modulus = {csys.modulus}")
print("secret key:")
print(f"p = {csys.p}")
print(f"q = {csys.q}")
print(f"kappa = {csys.kappa}")
print("computation test:")
print(f"x1 = {x1}, x2 = {x2}, x3 = {x3}, x4 = {x4}")
print(f"c1 = {c1}, c2 = {c2}, c3 = {c3}, c4 = {c4}")
print(f"y1 = {y1}, y2 = {y2}, y3 = {y3}, y4 = {y4}")