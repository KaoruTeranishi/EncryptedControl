from eclib import elgamal

key_length = 64
delta = 0.01
params, pk, sk = elgamal.keygen(key_length)

print("")
print("x =", x := 1.23)
print("pt_x = encode(x) =", pt_x := elgamal.encode(params, x, delta))
print("ct_x = encrypt(pt_x) =", ct_x := elgamal.encrypt(params, pk, pt_x))
print("decrypt(ct_x) =", elgamal.decrypt(params, sk, ct_x))
print("decode(pt_x) ≈", elgamal.decode(params, pt_x, delta))

print("")
print("y =", y := -4.56)
print("ct_y = encrypt(encode(y)) =", ct_y := elgamal.enc(params, pk, y, delta))
print("decode(decrypt(ct_y)) ≈", elgamal.dec(params, sk, ct_y, delta))

print("")
print("ct_z = mult(ct_x, ct_y) =", ct_z := elgamal.mult(params, ct_x, ct_y))
print("x * y ≈ decode(decrypt(ct_z)) =", elgamal.dec(params, sk, ct_z, delta**2))
print("")
