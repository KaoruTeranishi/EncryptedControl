from eclib import paillier

key_length = 64
delta = 0.01
params, pk, sk = paillier.keygen(key_length)

print("")
print("x =", x := 1.23)
print("pt_x = encode(x) =", pt_x := paillier.encode(params, x, delta))
print("ct_x = encrypt(pt_x) =", ct_x := paillier.encrypt(params, pk, pt_x))
print("decrypt(ct_x) =", paillier.decrypt(params, sk, ct_x))
print("decode(pt_x) ≈", paillier.decode(params, pt_x, delta))

print("")
print("y =", y := -4.56)
print("ct_y = encrypt(encode(y)) =", ct_y := paillier.enc(params, pk, y, delta))
print("decode(decrypt(ct_y)) ≈", paillier.dec(params, sk, ct_y, delta))

print("")
print("ct_z = add(ct_x, ct_y) =", ct_z := paillier.add(params, ct_x, ct_y))
print("x + y ≈ decode(decrypt(ct_z)) =", paillier.dec(params, sk, ct_z, delta))
print("")
