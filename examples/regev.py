from eclib import regev

security_params = (10, pow(2, 32), pow(2, 64), 3.2)
delta = 0.01
params, pk, sk = regev.keygen(*security_params)

print("")
print("x =", x := 1.23)
print("pt_x = encode(x) =", pt_x := regev.encode(params, x, delta))
print("ct_x = encrypt(pt_x) =\n", ct_x := regev.encrypt(params, pk, pt_x))
print("decrypt(ct_x) =", regev.decrypt(params, sk, ct_x))
print("decode(pt_x) ≈", regev.decode(params, pt_x, delta))

print("")
print("y =", y := -4.56)
print("ct_y = encrypt(encode(y)) =\n", ct_y := regev.enc(params, pk, y, delta))
print("decode(decrypt(ct_y)) ≈", regev.dec(params, sk, ct_y, delta))

print("")
print("ct_z = add(ct_x, ct_y) =\n", ct_z := regev.add(params, ct_x, ct_y))
print("x + y ≈ decode(decrypt(ct_z)) =", regev.dec(params, sk, ct_z, delta))
print("")
