#! /usr/bin/env python

from eclib.paillier import *

key_length = 64
params, pk, sk = keygen(key_length)

print('========== Paillier parameter ==========')
print(f'key length = {key_length} bit')
print(f'n = {params.n}')
print(f'n^2 = {params.n_square}')
print(f'g = {pk}')
print(f'lambda = {sk.lmd}')
print(f'mu = {sk.mu}')

n = 1
a1 = 9
b1 = 7
a2 = 9
b2 = 8
a = a1 + a2 + n
b = b1 + b2

x1 = 1.23
x2 = -4.56
x3 = x1 + x2
x4 = x1 * x2

c1 = enc(params, pk, x1, a, b, a1, b1)
c2 = enc(params, pk, x2, a, b, a2, b2)
c3 = add(params, c1, c2)
c4 = mult(params, encode(x1, a, b, a1, b1), c2)

y1 = dec_(params, sk, c1, a, b)
y2 = dec_(params, sk, c2, a, b)
y3 = dec_(params, sk, c3, a, b)
y4 = dec(params, sk, c4, a, b)

print('========== scalar & scalar ==========')
print(f'x1 = {x1:.2f}')
print(f'x2 = {x2:.2f}')
print(f'x3 = {x3:.2f}')
print(f'x4 = {x4:.2f}')
print(f'c1 = {c1}')
print(f'c2 = {c2}')
print(f'c3 = {c3}')
print(f'c4 = {c4}')
print(f'y1 = {y1:.2f}')
print(f'y2 = {y2:.2f}')
print(f'y3 = {y3:.2f}')
print(f'y4 = {y4:.2f}')

n = 3
a1 = 9
b1 = 7
a2 = 9
b2 = 8
a = a1 + a2 + n
b = b1 + b2

x1 = 1.23
x2 = [-0.12, -3.45, -6.78]
x3 = list(map(lambda a: x1 * a, x2))

c1 = enc(params, pk, x1, a, b, a1, b1)
c2 = enc(params, pk, x2, a, b, a2, b2)
c3 = mult(params, encode(x1, a, b, a1, b1), c2)

y1 = dec_(params, sk, c1, a, b)
y2 = dec_(params, sk, c2, a, b)
y3 = dec(params, sk, c3, a, b)

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

n = 3
a1 = 9
b1 = 7
a2 = 9
b2 = 8
a = a1 + a2 + n
b = b1 + b2

x1 = 1.23
x2 = [[0.12, 3.45, 6.78], [-0.12, -3.45, -6.78]]
x3 = [list(map(lambda a: x1 * a, x2[0])), list(map(lambda a: x1 * a, x2[1]))]

c1 = enc(params, pk, x1, a, b, a1, b1)
c2 = enc(params, pk, x2, a, b, a2, b2)
c3 = mult(params, encode(x1, a, b, a1, b1), c2)

y1 = dec_(params, sk, c1, a, b)
y2 = dec_(params, sk, c2, a, b)
y3 = dec(params, sk, c3, a, b)

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

n = 3
a1 = 9
b1 = 7
a2 = 9
b2 = 8
a = a1 + a2 + n
b = b1 + b2

x1 = [1.23, 4.56, 7.89]
x2 = [-0.12, -3.45, -6.78]
x3 = list(map(lambda a, b: a + b, x1, x2))
x4 = list(map(lambda a, b: a * b, x1, x2))
x5 = sum(x4)

c1 = enc(params, pk, x1, a, b, a1, b1)
c2 = enc(params, pk, x2, a, b, a2, b2)
c3 = add(params, c1, c2)
c4 = elementwise_mult(params, encode(x1, a, b, a1, b1), c2)
c5 = mult(params, encode(x1, a, b, a1, b1), c2)

y1 = dec_(params, sk, c1, a, b)
y2 = dec_(params, sk, c2, a, b)
y3 = dec_(params, sk, c3, a, b)
y4 = dec(params, sk, c4, a, b)
y5 = dec(params, sk, c5, a, b)

print('========== vector & vector ==========')
print(f'x1 = [{x1[0]:.2f}, {x1[1]:.2f}, {x1[2]:.2f}]')
print(f'x2 = [{x2[0]:.2f}, {x2[1]:.2f}, {x2[2]:.2f}]')
print(f'x3 = [{x3[0]:.2f}, {x3[1]:.2f}, {x3[2]:.2f}]')
print(f'x4 = [{x4[0]:.2f}, {x4[1]:.2f}, {x4[2]:.2f}]')
print(f'x5 = {x5:.2f}')
print(f'c1 = {c1}')
print(f'c2 = {c2}')
print(f'c3 = {c3}')
print(f'c4 = {c4}')
print(f'c5 = {c5}')
print(f'y1 = [{y1[0]:.2f}, {y1[1]:.2f}, {y1[2]:.2f}]')
print(f'y2 = [{y2[0]:.2f}, {y2[1]:.2f}, {y2[2]:.2f}]')
print(f'y3 = [{y3[0]:.2f}, {y3[1]:.2f}, {y3[2]:.2f}]')
print(f'y4 = [{y4[0]:.2f}, {y4[1]:.2f}, {y4[2]:.2f}]')
print(f'y5 = {y5:.2f}')

n = 3
a1 = 9
b1 = 7
a2 = 9
b2 = 8
a = a1 + a2 + n
b = b1 + b2

x1 = [[1.23, 4.56, 7.89], [-1.23, -4.56, -7.89]]
x2 = [0.12, 3.45, 6.78]
x3 = [list(map(lambda a, b: a + b, x1[0], x2)), list(map(lambda a, b: a + b, x1[1], x2))]
x4 = [list(map(lambda a, b: a * b, x1[0], x2)), list(map(lambda a, b: a * b, x1[1], x2))]
x5 = [sum(x4[0]), sum(x4[1])]

c1 = enc(params, pk, x1, a, b, a1, b1)
c2 = enc(params, pk, x2, a, b, a2, b2)
c3 = elementwise_add(params, c1, c2)
c4 = elementwise_mult(params, encode(x1, a, b, a1, b1), c2)
c5 = mult(params, encode(x1, a, b, a1, b1), c2)

y1 = dec_(params, sk, c1, a, b)
y2 = dec_(params, sk, c2, a, b)
y3 = dec_(params, sk, c3, a, b)
y4 = dec(params, sk, c4, a, b)
y5 = dec(params, sk, c5, a, b)

print('========== matrix & vector ==========')
print(f'x1 = [[{x1[0][0]:.2f}, {x1[0][1]:.2f}, {x1[0][2]:.2f}], [{x1[1][0]:.2f}, {x1[1][1]:.2f}, {x1[1][2]:.2f}]]')
print(f'x2 = [{x2[0]:.2f}, {x2[1]:.2f}, {x2[2]:.2f}]')
print(f'x3 = [[{x3[0][0]:.2f}, {x3[0][1]:.2f}, {x3[0][2]:.2f}], [{x3[1][0]:.2f}, {x3[1][1]:.2f}, {x3[1][2]:.2f}]]')
print(f'x4 = [[{x4[0][0]:.2f}, {x4[0][1]:.2f}, {x4[0][2]:.2f}], [{x4[1][0]:.2f}, {x4[1][1]:.2f}, {x4[1][2]:.2f}]]')
print(f'x5 = [{x5[0]:.2f}, {x5[1]:.2f}]')
print(f'c1 = {c1}')
print(f'c2 = {c2}')
print(f'c3 = {c3}')
print(f'c4 = {c4}')
print(f'c5 = {c5}')
print(f'y1 = [[{y1[0][0]:.2f}, {y1[0][1]:.2f}, {y1[0][2]:.2f}], [{y1[1][0]:.2f}, {y1[1][1]:.2f}, {y1[1][2]:.2f}]]')
print(f'y2 = [{y2[0]:.2f}, {y2[1]:.2f}, {y2[2]:.2f}]')
print(f'y3 = [[{y3[0][0]:.2f}, {y3[0][1]:.2f}, {y3[0][2]:.2f}], [{y3[1][0]:.2f}, {y3[1][1]:.2f}, {y3[1][2]:.2f}]]')
print(f'y4 = [[{y4[0][0]:.2f}, {y4[0][1]:.2f}, {y4[0][2]:.2f}], [{y4[1][0]:.2f}, {y4[1][1]:.2f}, {y4[1][2]:.2f}]]')
print(f'y5 = [{y5[0]:.2f}, {y5[1]:.2f}]')

n = 3
a1 = 9
b1 = 7
a2 = 9
b2 = 8
a = a1 + a2 + n
b = b1 + b2

x1 = [[1.23, 4.56, 7.89], [-1.23, -4.56, -7.89]]
x1_transpose = [list(x) for x in zip(*x1)]
x2 = [[0.12, 3.45, 6.78], [-0.12, -3.45, -6.78]]
x3 = [list(map(lambda a, b: a + b, x1[0], x2[0])), list(map(lambda a, b: a + b, x1[1], x2[1]))]
x4 = [list(map(lambda a, b: a * b, x1[0], x2[0])), list(map(lambda a, b: a * b, x1[1], x2[1]))]
x5 = [[0 for j in range(len(x2[0]))] for i in range(len(x1_transpose))]
for i in range(len(x5)):
    for j in range(len(x5[0])):
        for k in range(len(x1_transpose[0])):
            x5[i][j] += x1_transpose[i][k] * x2[k][j]

c1 = enc(params, pk, x1, a, b, a1, b1)
c2 = enc(params, pk, x2, a, b, a2, b2)
c3 = add(params, c1, c2)
c4 = elementwise_mult(params, encode(x1, a, b, a1, b1), c2)
c5 = mult(params, encode(x1_transpose, a, b, a1, b1), c2)

y1 = dec_(params, sk, c1, a, b)
y2 = dec_(params, sk, c2, a, b)
y3 = dec_(params, sk, c3, a, b)
y4 = dec(params, sk, c4, a, b)
y5 = dec(params, sk, c5, a, b)

print('========== matrix & matrix ==========')
print(f'x1 = [[{x1[0][0]:.2f}, {x1[0][1]:.2f}, {x1[0][2]:.2f}], [{x1[1][0]:.2f}, {x1[1][1]:.2f}, {x1[1][2]:.2f}]]')
print(f'x2 = [[{x2[0][0]:.2f}, {x2[0][1]:.2f}, {x2[0][2]:.2f}], [{x2[1][0]:.2f}, {x2[1][1]:.2f}, {x2[1][2]:.2f}]]')
print(f'x3 = [[{x3[0][0]:.2f}, {x3[0][1]:.2f}, {x3[0][2]:.2f}], [{x3[1][0]:.2f}, {x3[1][1]:.2f}, {x3[1][2]:.2f}]]')
print(f'x4 = [[{x4[0][0]:.2f}, {x4[0][1]:.2f}, {x4[0][2]:.2f}], [{x4[1][0]:.2f}, {x4[1][1]:.2f}, {x4[1][2]:.2f}]]')
print(f'x5 = [[{x5[0][0]:.2f}, {x5[0][1]:.2f}, {x5[0][2]:.2f}], [{x5[1][0]:.2f}, {x5[1][1]:.2f}, {x5[1][2]:.2f}], [{x5[2][0]:.2f}, {x5[2][1]:.2f}, {x5[2][2]:.2f}]]')
print(f'c1 = {c1}')
print(f'c2 = {c2}')
print(f'c3 = {c3}')
print(f'c4 = {c4}')
print(f'c5 = {c5}')
print(f'y1 = [[{y1[0][0]:.2f}, {y1[0][1]:.2f}, {y1[0][2]:.2f}], [{y1[1][0]:.2f}, {y1[1][1]:.2f}, {y1[1][2]:.2f}]]')
print(f'y2 = [[{y2[0][0]:.2f}, {y2[0][1]:.2f}, {y2[0][2]:.2f}], [{y2[1][0]:.2f}, {y2[1][1]:.2f}, {y2[1][2]:.2f}]]')
print(f'y3 = [[{y3[0][0]:.2f}, {y3[0][1]:.2f}, {y3[0][2]:.2f}], [{y3[1][0]:.2f}, {y3[1][1]:.2f}, {y3[1][2]:.2f}]]')
print(f'y4 = [[{y4[0][0]:.2f}, {y4[0][1]:.2f}, {y4[0][2]:.2f}], [{y4[1][0]:.2f}, {y4[1][1]:.2f}, {y4[1][2]:.2f}]]')
print(f'y5 = [[{y5[0][0]:.2f}, {y5[0][1]:.2f}, {y5[0][2]:.2f}], [{y5[1][0]:.2f}, {y5[1][1]:.2f}, {y5[1][2]:.2f}], [{y5[2][0]:.2f}, {y5[2][1]:.2f}, {y5[2][2]:.2f}]]')
