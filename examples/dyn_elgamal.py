from eclib.dyn_elgamal import *
import numpy as np

bit_length = 64
delta = 1e-4

params, pk, sk = keygen(bit_length)

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
ct_v1 = enc(params, pk, v1, delta)
ct_m1 = enc(params, pk, m1, delta)

pk, sk, token = update_key(params, pk, sk)

ct_x2 = enc(params, pk, x2, delta)
ct_v2 = enc(params, pk, v2, delta)
ct_m2 = enc(params, pk, m2, delta)

ct_x1 = update_ct(params, ct_x1, token)
ct_v1 = update_ct(params, ct_v1, token)
ct_m1 = update_ct(params, ct_m1, token)

print('')
print('multiplication')
print(f'x1 * x2 = {x1 * x2}')
print(f'enc: {dec(params, sk, mult(params, ct_x1, ct_x2), delta ** 2)}')
print(f'x1 * v2 = {x1 * v2}')
print(f'enc: {dec(params, sk, mult(params, ct_x1, ct_v2), delta ** 2)}')
print(f'x1 * m2 = \n{x1 * m2}')
print(f'enc: \n{dec(params, sk, mult(params, ct_x1, ct_m2), delta ** 2)}')
print(f'v1 * v2 = {v1 @ v2}')
print(f'enc: {dec_add(params, sk, mult(params, ct_v1, ct_v2), delta ** 2)}')
print(f'm1 * v2 = {m1 @ v2}')
print(f'enc: {dec_add(params, sk, mult(params, ct_m1, ct_v2), delta ** 2)}')
print(f'v1 .* v2 = {v1 * v2}')
print(f'enc: {dec(params, sk, mult(params, ct_v1, ct_v2), delta ** 2)}')
print(f'm1 .* v2 = \n{m1 * v2}')
print(f'enc: \n{dec(params, sk, mult(params, ct_m1, ct_v2), delta ** 2)}')
print(f'm1 .* m2 = \n{m1 * m2}')
print(f'enc: \n{dec(params, sk, mult(params, ct_m1, ct_m2), delta ** 2)}')
print('')
