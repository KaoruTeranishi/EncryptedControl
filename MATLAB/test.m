clear

bitLength = 20;
cryptosystem = ElGamal(bitLength);

gamma = 1e2;

x1 = 1.23;
x2 = -4.56;
x3 = x1*x2;

c1 = cryptosystem.Enc(x1,gamma);
c2 = cryptosystem.Enc(x2,gamma);

c3 = cryptosystem.Mult(c1,c2);

y1 = cryptosystem.Dec(c1,gamma);
y2 = cryptosystem.Dec(c2,gamma);
y3 = cryptosystem.Dec(c3,gamma^2);