from eclib import gsw_lwe

security_params = (10, pow(2, 20), pow(2, 32), 3.2)
delta = 0.01
params, pk, sk = gsw_lwe.keygen(*security_params)

print("")
print("x =", x := 1.23)
print("pt_x = encode(x) =", pt_x := gsw_lwe.encode(params, x, delta))
print("ct_x = encrypt(pt_x) =\n", ct_x := gsw_lwe.encrypt(params, pk, pt_x))
print("decrypt(ct_x) =", gsw_lwe.decrypt(params, sk, ct_x))
print("decode(pt_x) ≈", gsw_lwe.decode(params, pt_x, delta))

print(
    "gsw_ct_x = encryptGSW(pt_x) =\n",
    gsw_ct_x := gsw_lwe.encrypt_gsw(params, pk, pt_x),
)
print("decrypt(gsw_ct_x) =", gsw_lwe.decrypt_gsw(params, sk, gsw_ct_x))

print("")
print("y =", y := -4.56)
print("ct_y = encrypt(encode(y)) =\n", ct_y := gsw_lwe.enc(params, pk, y, delta))
print("decode(decrypt(ct_y)) ≈", gsw_lwe.dec(params, sk, ct_y, delta))

print("")
print("ct_z = add(ct_x, ct_y) =\n", ct_z := gsw_lwe.add(params, ct_x, ct_y))
print("x + y ≈ decode(decrypt(ct_z)) =", gsw_lwe.dec(params, sk, ct_z, delta))

print("")
print(
    "ct_w = mult(gsw_ct_x, ct_y) =\n",
    ct_w := gsw_lwe.mult(params, gsw_ct_x, ct_y),
)
print("x * y ≈ decode(decrypt(ct_w)) =", gsw_lwe.dec(params, sk, ct_w, delta**2))
print("")
