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

**MATLAB**
- MATLAB R2019a or higher
- Control System Toolbox

**Python**
- Python 3.7.8
- numpy
- matplotlib
- python-control
- slycot
- pycrypto

# How to use
1. Clone this repository. `git clone https://github.com/KaoruTeranishi/EncryptedControl.git`
1. Install the requirements.
1. Execute codes in each directory.

# Encrypted control
A concept of controller encryption is first proposed by Kiminao Kogiso and Takahiro Fujita in 2015 IEEE 54th Conference on Decision and Control (CDC).
An encrypted controller directly decides control inputs from encrypted mesurements and encrypted controller parameters without decryption.
For more details, please see [here](https://en.kimilab.tokyo/contents/31).

# ElGamal encryption
[ElGamal encryption](https://en.wikipedia.org/wiki/ElGamal_encryption) is multiplicative homomorphic encryption, which allows to compute multiplication in ciphertext.

**References**
1. T. Elgamal, "A public key cryptosystem and a signature scheme based on discrete logarithms," in CRYPTO, vol. 196, 1984, pp. 10–18.
1. K. Kogiso and T. Fujita, "Cyber-security enhancement of networked control systems using homomorphic encryption," in IEEE Conference on Decision and Control, 2015, pp. 6836–6843.
1. K. Teranishi, N. Shimada, and K. Kogiso, "Stability analysis and dynamic quantizer for controller encryption," in IEEE Conference on Decision and Control, 2019, pp. 7184-7189.
1. K. Teranishi, J. Ueda, and K. Kogiso, "Event-triggered approach to increasing sampling period of encrypted control systems," IFAC World Congress, 2020.
1. K. Teranishi, N. Shimada, and K. Kogiso, "Stability-guaranteed dynamic ElGamal cryptosystem for encrypted control systems," IET Control Theory & Applications. (in press)

# Paillier encryption
[Paillier encryption](https://en.wikipedia.org/wiki/Paillier_cryptosystem) is additive homomorphic encryption, which allows to compute addition in ciphertext.

**References**
1. P. Paillier, "Public-key cryptosystems based on composite degree residuosity classes," in EUROCRYPT, 1999, pp. 223–238.
1. F. Farokhi, I. Shames, and N. Batterham, "Secure and private control using semi-homomorphic encryption," Control Engineering Practice, vol. 67, pp. 13–20, 2017.

# Somewhat homomorphic encryption
Somewhat homomorphic encryption allows to compute limited number of multiplication and addition in ciphertext.
This library employs Dyer et al.'s somewhat homomorphic encrypion.

**References**
1. J. Dyer, M. Dyer, and J. Xu, "Practical homomorphic encryption over the integers for secure computation in the cloud," International Journal of Information Security, vol. 18, no. 5, pp. 549-579, 2019.
1. K. Teranishi, K. Kogiso, and J. Ueda, "Encrypted feedback linearization and motion control for manipulator with somewhat homomorphic encryption," IEEE/ASME International Conference on Advanced Intelligent Mechatronics, 2020, pp.613-618.

# License
BSD License 2.0

# Author
Kaoru Teranishi
- E-mail: teranishi (at) uec.ac.jp
- Profile: [https://kaoruteranishi.xyz](https://kaoruteranishi.xyz)
