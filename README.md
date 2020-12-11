<img src="https://github.com/KaoruTeranishi/EncryptedControl/blob/master/logo.png?raw=true" align="center" width="400" alt="header pic"/>

# ECLib

This is a Python library for numerical simulation of encrypted control.

# Encrypted control

Cybersecurity is a critical issue of networked control systems in a modern society.
Encrypted control is a novel concept of control using cryptographic tools for secure computation, such as homomorphic encryption and secret sharing.
A control input in an encrypted control system is computed with encrypted data to improve the confidentiality of control systems.
ECLib helps researchers and students to implement their new idea of encrypted control.

# Install

You can use `pip`.

`pip install eclib`

You may have to install slycot for running example codes.

# Examples

| Filename                               | Encryption      | Controller                     | Description                                                     |
| -------------------------------------- | --------------- | ------------------------------ | --------------------------------------------------------------- |
| `ElGamal/test.py` 				     | ElGamal         |                                | The example of ElGamal encryption                               |
| `ElGamal/enc_state_feedback.py`        | ElGamal         | State feedback                 | Encrypted state-feedback                                        |
| `ElGamal/dynqtz_enc_state_feedback.py` | ElGamal         | State feedback                 | Encrypted state-feedback with dynamic quantizer                 |
| `ElGamal/dynqtz_enc_observer.py`       | ElGamal         | Output feedback                | Encrypted output-feedback with dynamic quantizer                |
| `ElGamal/evt_state_feedback.py`        | ElGamal         | Event-triggered state feedback | Encrypted event-triggered state-feedback with dynamic quantizer |
| `ElGamal/dynenc_state_feedback.py`     | Dynamic ElGamal | State feedback                 | Encrypted state-feedback using dynamic-key encryption           |
| `Paillier/test.py`                     | Paillier        |                                | The example of Paillier encryption                              |
| `Paillier/enc_state_feedback.py`       | Paillier        | State feedback                 | Encrypted state-feedback                                        |

# References

1. T. Elgamal, "A public key cryptosystem and a signature scheme based on discrete logarithms," in IEEE Transactions on Information Theory, vol. 31, no. 4, pp. 469-472, 1985.
1. P. Paillier, "Public-key cryptosystems based on composite degree residuosity classes," in EUROCRYPT, 1999, pp. 223–238.
1. K. Kogiso and T. Fujita, "Cyber-security enhancement of networked control systems using homomorphic encryption," in IEEE Conference on Decision and Control, 2015, pp. 6836–6843.
1. F. Farokhi, I. Shames, and N. Batterham, "Secure and private control using semi-homomorphic encryption," Control Engineering Practice, vol. 67, pp. 13–20, 2017.

# License

BSD License 2.0

# Author

Kaoru Teranishi
- E-mail: teranishi (at) uec.ac.jp
- Homepage: [https://kaoruteranishi.xyz](https://kaoruteranishi.xyz)
