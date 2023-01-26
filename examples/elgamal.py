from eclib.elgamal import *

key_length = 20
params, pk, sk = keygen(key_length)

print('ElGamal cryptosystem:')
print(f'\u03BB = {key_length}')
print(f'p = {params.p}')
print(f'q = {params.q}')
print(f'g = {params.g}')
print(f'h = {pk}')
print(f's = {sk}')

delta = 0.01
x1 = 1.23
m1 = encode(params, x1, delta)
c1 = encrypt(params, pk, m1)
n1 = decrypt(params, sk, c1)
y1 = decode(params, n1, delta)

print(f'x1 = {x1}')
print(f'm1 = Ecd(x1) = {m1}')
print(f'c1 = Enc(m1) = {c1}')
print(f'n1 = Dec(c1) = {n1}')
print(f'y1 = Dcd(n1) = {y1}')

x2 = -4.56
c2 = enc(params, pk, x2, delta)
y2 = dec(params, sk, c2, delta)

print(f'x2 = {x2}')
print(f'c2 = Enc(Ecd(x2)) = {c2}')
print(f'y2 = Dcd(Dec(c2)) = {y2}')

x3 = x1 * x2
c3 = mult(params, c1, c2)
y3 = dec(params, sk, c3, delta ** 2)

print(f'x3 = x1 * x2 = {x3}')
print(f'c3 = Mult(c1, c2) = {c3}')
print(f'y3 = Dcd(Dec(c3)) = {y3}')