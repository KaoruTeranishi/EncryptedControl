from eclib.gsw import *
import numpy as np

n = 5
q = pow(2, 48)
sigma = 3.2
delta = 1e-6

params, pk, sk = keygen(n, q, sigma)

x1 = 1.23
x2 = -2.34
v1 = np.array([1.23, 2.34])
v2 = np.array([-2.34, -3.45])
m1 = np.array([[1.23, 2.34], [2.34, 3.45]])
m2 = np.array([[-2.34, -3.45], [-4.56, -5.67]])

pt_x1 = encode(params, x1, delta)
pt_v1 = encode(params, v1, delta)
pt_m1 = encode(params, m1, delta)

ct_x1 = enc(params, pk, x1, delta)
ct_x2 = enc(params, pk, x2, delta)
ct_v1 = enc(params, pk, v1, delta)
ct_v2 = enc(params, pk, v2, delta)
ct_m1 = enc(params, pk, m1, delta)
ct_m2 = enc(params, pk, m2, delta)

print('')
print('addition')
print(f'x1 + x2 = {x1 + x2}')
print(f'enc: {dec(params, sk, add(params, ct_x1, ct_x2), delta)}')
print(f'v1 + v2 = {v1 + v2}')
print(f'enc: {dec(params, sk, add(params, ct_v1, ct_v2), delta)}')
print(f'm1 + m2 = \n{m1 + m2}')
print(f'enc: \n{dec(params, sk, add(params, ct_m1, ct_m2), delta)}')
print('')

print('elementwise addition')
print(f'm1 + v2 = \n{m1 + v2}')
print(f'enc: \n{dec(params, sk, elementwise_add(params, ct_m1, ct_v2), delta)}')
print('')

print('multiplication')
print(f'x1 * x2 = {x1 * x2}')
print(f'enc: {dec(params, sk, mult(params, ct_x1, ct_x2), delta ** 2)}')
print(f'x1 * v2 = {x1 * v2}')
print(f'enc: {dec(params, sk, mult(params, ct_x1, ct_v2), delta ** 2)}')
print(f'x1 * m2 = \n{x1 * m2}')
print(f'enc: \n{dec(params, sk, mult(params, ct_x1, ct_m2), delta ** 2)}')
print(f'v1 * v2 = {v1 @ v2}')
print(f'enc: {dec(params, sk, mult(params, ct_v1, ct_v2), delta ** 2)}')
print(f'm1 * v2 = {m1 @ v2}')
print(f'enc: {dec(params, sk, mult(params, ct_m1, ct_v2), delta ** 2)}')
print(f'm1 * m2 = \n{m1 @ m2}')
print(f'enc: \n{dec(params, sk, mult(params, ct_m1, ct_m2), delta ** 2)}')
print('')

print('elementwise multiplication')
print(f'v1 .* v2 = {v1 * v2}')
print(f'enc: {dec(params, sk, elementwise_mult(params, ct_v1, ct_v2), delta ** 2)}')
print(f'm1 .* v2 = \n{m1 * v2}')
print(f'enc: \n{dec(params, sk, elementwise_mult(params, ct_m1, ct_v2), delta ** 2)}')
print(f'm1 .* m2 = \n{m1 * m2}')
print(f'enc: \n{dec(params, sk, elementwise_mult(params, ct_m1, ct_m2), delta ** 2)}')
print('')

print('integer multiplication')
print(f'x1 * x2 = {x1 * x2}')
print(f'enc: {dec(params, sk, int_mult(params, pt_x1, ct_x2), delta ** 2)}')
print(f'x1 * v2 = {x1 * v2}')
print(f'enc: {dec(params, sk, int_mult(params, pt_x1, ct_v2), delta ** 2)}')
print(f'x1 * m2 = \n{x1 * m2}')
print(f'enc: \n{dec(params, sk, int_mult(params, pt_x1, ct_m2), delta ** 2)}')
print(f'v1 * v2 = {v1 @ v2}')
print(f'enc: {dec(params, sk, int_mult(params, pt_v1, ct_v2), delta ** 2)}')
print(f'm1 * v2 = {m1 @ v2}')
print(f'enc: {dec(params, sk, int_mult(params, pt_m1, ct_v2), delta ** 2)}')
print(f'm1 * m2 = \n{m1 @ m2}')
print(f'enc: \n{dec(params, sk, int_mult(params, pt_m1, ct_m2), delta ** 2)}')
print('')

print('elementwise integer multiplication')
print(f'v1 .* v2 = {v1 * v2}')
print(f'enc: {dec(params, sk, elementwise_int_mult(params, pt_v1, ct_v2), delta ** 2)}')
print(f'm1 .* v2 = \n{m1 * v2}')
print(f'enc: \n{dec(params, sk, elementwise_int_mult(params, pt_m1, ct_v2), delta ** 2)}')
print(f'm1 .* m2 = \n{m1 * m2}')
print(f'enc: \n{dec(params, sk, elementwise_int_mult(params, pt_m1, ct_m2), delta ** 2)}')
print('')
