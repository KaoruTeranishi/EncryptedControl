#! /usr/bin/env python

from eclib.elgamal import *

key_length = 20
params, pk, sk = keygen(key_length)
delta = 1e-2

print('========== ElGamal parameter ==========')
print(f'key length = {key_length} bit')
print(f'p = {params.p}')
print(f'q = {params.q}')
print(f'g = {params.g}')
print(f'h = {pk}')
print(f's = {sk}')

x1 = 1.23
x2 = -4.56
x3 = x1 * x2

c1 = enc(params, pk, x1, delta)
c2 = enc(params, pk, x2, delta)
c3 = mult(params, c1, c2)

y1 = dec(params, sk, c1, delta)
y2 = dec(params, sk, c2, delta)
y3 = dec(params, sk, c3, delta ** 2)

print('========== scalar & scalar ==========')
print(f'x1 = {x1:.2f}')
print(f'x2 = {x2:.2f}')
print(f'x3 = {x3:.2f}')
print(f'c1 = {c1}')
print(f'c2 = {c2}')
print(f'c3 = {c3}')
print(f'y1 = {y1:.2f}')
print(f'y2 = {y2:.2f}')
print(f'y3 = {y3:.2f}')

x1 = 1.23
x2 = [-0.12, -3.45, -6.78]
x3 = list(map(lambda a: x1 * a, x2))

c1 = enc(params, pk, x1, delta)
c2 = enc(params, pk, x2, delta)
c3 = mult(params, c1, c2)

y1 = dec(params, sk, c1, delta)
y2 = dec(params, sk, c2, delta)
y3 = dec(params, sk, c3, delta ** 2)

print('========== scalar & vector ==========')
print(f'x1 = {x1:.2f}')
print(f'x2 = [{x2[0]:.2f}, {x2[1]:.2f}, {x2[2]:.2f}]')
print(f'x3 = [{x3[0]:.2f}, {x3[1]:.2f}, {x3[2]:.2f}]')
print(f'c1 = {c1}')
print(f'c2 = {c2}')
print(f'c3 = {c3}')
print(f'y1 = {y1:.2f}')
print(f'y2 = [{y2[0]:.2f}, {y2[1]:.2f}, {y2[2]:.2f}]')
print(f'y3 = [{y3[0]:.2f}, {y3[1]:.2f}, {y3[2]:.2f}]')

x1 = 1.23
x2 = [[0.12, 3.45, 6.78], [-0.12, -3.45, -6.78]]
x3 = [list(map(lambda a: x1 * a, x2[0])), list(map(lambda a: x1 * a, x2[1]))]

c1 = enc(params, pk, x1, delta)
c2 = enc(params, pk, x2, delta)
c3 = mult(params, c1, c2)

y1 = dec(params, sk, c1, delta)
y2 = dec(params, sk, c2, delta)
y3 = dec(params, sk, c3, delta ** 2)

print('========== scalar & matrix ==========')
print(f'x1 = {x1:.2f}')
print(f'x2 = [[{x2[0][0]:.2f}, {x2[0][1]:.2f}, {x2[0][2]:.2f}], [{x2[1][0]:.2f}, {x2[1][1]:.2f}, {x2[1][2]:.2f}]]')
print(f'x3 = [[{x3[0][0]:.2f}, {x3[0][1]:.2f}, {x3[0][2]:.2f}], [{x3[1][0]:.2f}, {x3[1][1]:.2f}, {x3[1][2]:.2f}]]')
print(f'c1 = {c1}')
print(f'c2 = {c2}')
print(f'c3 = {c3}')
print(f'y1 = {y1:.2f}')
print(f'y2 = [[{y2[0][0]:.2f}, {y2[0][1]:.2f}, {y2[0][2]:.2f}], [{y2[1][0]:.2f}, {y2[1][1]:.2f}, {y2[1][2]:.2f}]]')
print(f'y3 = [[{y3[0][0]:.2f}, {y3[0][1]:.2f}, {y3[0][2]:.2f}], [{y3[1][0]:.2f}, {y3[1][1]:.2f}, {y3[1][2]:.2f}]]')

x1 = [1.23, 4.56, 7.89]
x2 = [-0.12, -3.45, -6.78]
x3 = list(map(lambda a, b: a * b, x1, x2))

c1 = enc(params, pk, x1, delta)
c2 = enc(params, pk, x2, delta)
c3 = mult(params, c1, c2)

y1 = dec(params, sk, c1, delta)
y2 = dec(params, sk, c2, delta)
y3 = dec(params, sk, c3, delta ** 2)

print('========== vector & vector ==========')
print(f'x1 = [{x1[0]:.2f}, {x1[1]:.2f}, {x1[2]:.2f}]')
print(f'x2 = [{x2[0]:.2f}, {x2[1]:.2f}, {x2[2]:.2f}]')
print(f'x3 = [{x3[0]:.2f}, {x3[1]:.2f}, {x3[2]:.2f}]')
print(f'c1 = {c1}')
print(f'c2 = {c2}')
print(f'c3 = {c3}')
print(f'y1 = [{y1[0]:.2f}, {y1[1]:.2f}, {y1[2]:.2f}]')
print(f'y2 = [{y2[0]:.2f}, {y2[1]:.2f}, {y2[2]:.2f}]')
print(f'y3 = [{y3[0]:.2f}, {y3[1]:.2f}, {y3[2]:.2f}]')

x1 = [[1.23, 4.56, 7.89], [-1.23, -4.56, -7.89]]
x2 = [0.12, 3.45, 6.78]
x3 = [list(map(lambda a, b: a * b, x1[0], x2)), list(map(lambda a, b: a * b, x1[1], x2))]

c1 = enc(params, pk, x1, delta)
c2 = enc(params, pk, x2, delta)
c3 = mult(params, c1, c2)

y1 = dec(params, sk, c1, delta)
y2 = dec(params, sk, c2, delta)
y3 = dec(params, sk, c3, delta ** 2)

print('========== matrix & vector ==========')
print(f'x1 = [[{x1[0][0]:.2f}, {x1[0][1]:.2f}, {x1[0][2]:.2f}], [{x1[1][0]:.2f}, {x1[1][1]:.2f}, {x1[1][2]:.2f}]]')
print(f'x2 = [{x2[0]:.2f}, {x2[1]:.2f}, {x2[2]:.2f}]')
print(f'x3 = [[{x3[0][0]:.2f}, {x3[0][1]:.2f}, {x3[0][2]:.2f}], [{x3[1][0]:.2f}, {x3[1][1]:.2f}, {x3[1][2]:.2f}]]')
print(f'c1 = {c1}')
print(f'c2 = {c2}')
print(f'c3 = {c3}')
print(f'y1 = [[{y1[0][0]:.2f}, {y1[0][1]:.2f}, {y1[0][2]:.2f}], [{y1[1][0]:.2f}, {y1[1][1]:.2f}, {y1[1][2]:.2f}]]')
print(f'y2 = [{y2[0]:.2f}, {y2[1]:.2f}, {y2[2]:.2f}]')
print(f'y3 = [[{y3[0][0]:.2f}, {y3[0][1]:.2f}, {y3[0][2]:.2f}], [{y3[1][0]:.2f}, {y3[1][1]:.2f}, {y3[1][2]:.2f}]]')

x1 = [[1.23, 4.56, 7.89], [-1.23, -4.56, -7.89]]
x2 = [[0.12, 3.45, 6.78], [-0.12, -3.45, -6.78]]
x3 = [list(map(lambda a, b: a * b, x1[0], x2[0])), list(map(lambda a, b: a * b, x1[1], x2[1]))]

c1 = enc(params, pk, x1, delta)
c2 = enc(params, pk, x2, delta)
c3 = mult(params, c1, c2)

y1 = dec(params, sk, c1, delta)
y2 = dec(params, sk, c2, delta)
y3 = dec(params, sk, c3, delta ** 2)

print('========== matrix & matrix ==========')
print(f'x1 = [[{x1[0][0]:.2f}, {x1[0][1]:.2f}, {x1[0][2]:.2f}], [{x1[1][0]:.2f}, {x1[1][1]:.2f}, {x1[1][2]:.2f}]]')
print(f'x2 = [[{x2[0][0]:.2f}, {x2[0][1]:.2f}, {x2[0][2]:.2f}], [{x2[1][0]:.2f}, {x2[1][1]:.2f}, {x2[1][2]:.2f}]]')
print(f'x3 = [[{x3[0][0]:.2f}, {x3[0][1]:.2f}, {x3[0][2]:.2f}], [{x3[1][0]:.2f}, {x3[1][1]:.2f}, {x3[1][2]:.2f}]]')
print(f'c1 = {c1}')
print(f'c2 = {c2}')
print(f'c3 = {c3}')
print(f'y1 = [[{y1[0][0]:.2f}, {y1[0][1]:.2f}, {y1[0][2]:.2f}], [{y1[1][0]:.2f}, {y1[1][1]:.2f}, {y1[1][2]:.2f}]]')
print(f'y2 = [[{y2[0][0]:.2f}, {y2[0][1]:.2f}, {y2[0][2]:.2f}], [{y2[1][0]:.2f}, {y2[1][1]:.2f}, {y2[1][2]:.2f}]]')
print(f'y3 = [[{y3[0][0]:.2f}, {y3[0][1]:.2f}, {y3[0][2]:.2f}], [{y3[1][0]:.2f}, {y3[1][1]:.2f}, {y3[1][2]:.2f}]]')
