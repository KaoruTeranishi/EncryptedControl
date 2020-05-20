<img src="https://github.com/KaoruTeranishi/EncryptedControl/blob/master/logo.png?raw=true" align="center" width="400" alt="header pic"/>

# Table of contents
- [What is this?](#what-is-this)
- [Requirements](#requirements)
- [How to use](#how-to-use)
- [Encrypted control](#encrypted-control)
- [ElGamal encryption](#elgamal-encryption)
- [Paillier encryption](#paillier-encryption)
- [Somewhat homomorphic encryption](#somewhat-homomorphic-encryption)
- [References](#references)
- [License](#license)
- [Author](#author)

# What is this?
This is Python and MATLAB libraries for encrypted control.

# Requirements
- Python 3.7.x
- numpy
- matplotlib
- python-control
- slycot
- pycrypto

# How to use
1. Clone this repository. `git clone https://github.com/KaoruTeranishi/EncryptedControl.git`
2. Install the above requirements.
3. Execute codes in each directory.

# Encrypted control
A concept of controller encryption is first proposed by Kiminao Kogiso and Takahiro Fujita in 2015 IEEE 54th Conference on Decision and Control (CDC).
An encrypted controller decides control inputs from encrypted mesurements and encrypted controller parameters without decryption.
For more details, please see [here](https://en.kimilab.tokyo/contents/31).

# ElGamal encryption
[ElGamal encryption](https://en.wikipedia.org/wiki/ElGamal_encryption), multiplicative homomorphic encryption, allows to execute multiplication in ciphertext.
The original encrypted control system uses RSA or ElGamal encryption.

# Paillier encryption
[Paillier encryption](https://en.wikipedia.org/wiki/Paillier_cryptosystem), additive homomorphic encryption, allows to execute addition in ciphertext.
This library employs Farokhi et al.'s algorithms.

# Somewhat homomorphic encryption
Somewhat homomorphic encryption allows to execute limited number of multiplication and addition in ciphertext.
This library employs Dyer et al.'s somewhat homomorphic encrypion.

# References
1. K. Kogiso and T. Fujita, "Cyber-security enhancement of networked control systems using homomorphic encryption," in IEEE Conference on Decision and Control, 2015, pp. 6836–6843.
2. F. Farokhi, I. Shames, and N. Batterham, "Secure and private control using semi-homomorphic encryption," Control Engineering Practice, vol. 67, pp. 13–20, 2017.
3. T. Elgamal, "A public key cryptosystem and a signature scheme based on discrete logarithms," in CRYPTO, vol. 196, 1984, pp. 10–18.
4. P. Paillier, "Public-key cryptosystems based on composite degree residuosity classes," in EUROCRYPT, 1999, pp. 223–238.
5. J. Dyer, M. Dyer, and J. Xu, "Practical homomorphic encryption over the integers for secure computation in the cloud," International Journal of Information Security, vol. 18, no. 5, pp. 549-579, 2019.

# License
BSD License 2.0

# Author
Kaoru Teranishi
- E-mail: teranishi (at) uec.ac.jp
- Profile: [https://kaoruteranishi.xyz](https://kaoruteranishi.xyz)
