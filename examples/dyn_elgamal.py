from eclib import dyn_elgamal

key_length = 64
delta = 0.01
params, pk, sk = dyn_elgamal.keygen(key_length)

print("")
print("pk =", pk)
print("sk =", sk)

print("")
print("x =", x := 1.23)
print("pt_x = encode(x) =", pt_x := dyn_elgamal.encode(params, x, delta))
print("ct_x = encrypt(pt_x) =", ct_x := dyn_elgamal.encrypt(params, pk, pt_x))
print("decrypt(ct_x) =", dyn_elgamal.decrypt(params, sk, ct_x))
print("decode(pt_x) ≈", dyn_elgamal.decode(params, pt_x, delta))

pk, sk, token = dyn_elgamal.update_key(params, pk, sk)
ct_x = dyn_elgamal.update_ct(params, ct_x, token)

print("")
print("pk =", pk)
print("sk =", sk)
print("ct_x =", ct_x)

print("")
print("y =", y := -4.56)
print("ct_y = encrypt(encode(y)) =", ct_y := dyn_elgamal.enc(params, pk, y, delta))
print("decode(decrypt(ct_y)) ≈", dyn_elgamal.dec(params, sk, ct_y, delta))

print("")
print("ct_z = mult(ct_x, ct_y) =", ct_z := dyn_elgamal.mult(params, ct_x, ct_y))
print("x * y ≈ decode(decrypt(ct_z)) =", dyn_elgamal.dec(params, sk, ct_z, delta**2))
print("")
