from eclib import gsw

security_params = (10, pow(2, 32), 3.2)
delta = 0.01
params, pk, sk = gsw.keygen(*security_params)

print("")
print("x =", x := 1.23)
print("pt_x = encode(x) =", pt_x := gsw.encode(params, x, delta))
print("ct_x = encrypt(pt_x) =\n", ct_x := gsw.encrypt(params, pk, pt_x))
print("decrypt(ct_x) =", gsw.decrypt(params, sk, ct_x))
print("decode(pt_x) ≈", gsw.decode(params, pt_x, delta))

print("")
print("y =", y := -4.56)
print("ct_y = encrypt(encode(y)) =\n", ct_y := gsw.enc(params, pk, y, delta))
print("decode(decrypt(ct_y)) ≈", gsw.dec(params, sk, ct_y, delta))

print("")
print("ct_z = add(ct_x, ct_y) =\n", ct_z := gsw.add(params, ct_x, ct_y))
print("x + y ≈ decode(decrypt(ct_z)) =", gsw.dec(params, sk, ct_z, delta))

print("")
print("ct_w = mult(ct_x, ct_y) =\n", ct_w := gsw.mult(params, ct_x, ct_y))
print("x * y ≈ decode(decrypt(ct_w)) =", gsw.dec(params, sk, ct_w, delta**2))
print("")
