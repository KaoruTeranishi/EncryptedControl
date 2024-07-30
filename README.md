<img src="https://github.com/KaoruTeranishi/EncryptedControl/blob/master/logo.png?raw=true" align="center" width="400" alt="header pic"/>

# ECLib

ECLib is an easy-to-use open-source homomorphic encryption library for encrypted control.

## Supported encryption schemes

- [ElGamal](https://en.wikipedia.org/wiki/ElGamal_encryption)
- [Dynamic-key ElGamal](https://arxiv.org/abs/2104.12436)
- [Paillier](https://en.wikipedia.org/wiki/Paillier_cryptosystem)
- [Regev (LWE)](https://en.wikipedia.org/wiki/Learning_with_errors)
- [GSW](https://eprint.iacr.org/2013/340)
- [GSW-LWE](https://eprint.iacr.org/2016/870)

## Requirements

- [Python 3.12.x](https://www.python.org/)
- [NumPy](https://numpy.org/doc/stable/index.html)
- [Matplotlib](https://matplotlib.org/)
- [python-control](https://python-control.readthedocs.io/en/0.10.0/)

# Getting Started

1. Install ECLib.

    `pip install eclib`

2. Run example scripts.

# Examples

- elgamal.py

    Basics of ElGamal encryption.

- dyn_elgamal.py

    Basics of Dynamic-key ElGamal encryption.

- paillier.py

    Basics of Paillier encryption.

- regev.py

    Basics of Regev (LWE) encryption.

- gsw.py

    Basics of GSW encryption.

- gsw_lwe.py

    Basics of GSW-LWE encryption.

- state_feedback.py

    Comparison of plant inputs and states between unencrypted and encrypted state-feedback controls.
    To run this script, you must specify an encryption scheme for encrypting a state-feedback controller, e.g., `python state_feedback.py elgamal`.

- pi.py

    Comparison of plant inputs and outputs between unencrypted and encrypted PI controls.
    Similar to state_feedback.py, you must specify an encryption scheme for encrypting a PI controller, e.g., `python pi.py paillier`.

# License

MIT license

# Author

Kaoru Teranishi
- E-mail: kaoruteranishi1005 (at) gmail.com
- Homepage: [https://kaoruteranishi.xyz](https://kaoruteranishi.xyz)
