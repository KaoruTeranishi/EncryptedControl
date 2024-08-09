# What is ECLib?

![logo](../_static/images/logo.png)

ECLib is an open source homomorphic encryption library for encrypted control.
This library is written in Python for ease of use and is designed to provide a high-level interface for users to develop encrypted control systems.
ECLib implements intentionally unoptimized cryptographic algorithms so that their behavior can be easily understood and adapted to accelerate research, education, and prototyping in the area of encrypted control.


## Background

Cyber-physical systems (CPS) integrate the digital and physical worlds to improve decision making in  various sectors.
However, CPS often face security and privacy issues due to insecure communication channels and untrusted third parties.
An adversary who learns confidential information from CPS can launch sophisticated and undetectable attacks.
Although encryption is a common approach to achieve secure communication in CPS, traditional encryption schemes still make data processing insecure because they expose data during computation processes.


## Homomorphic encryption

Homomorphic encryption is a form of encryption that allows direct computation on encrypted data without the need for decryption.
This property is particularly useful for performing computations on sensitive data while preserving data privacy when outsourcing computation to untrusted third parties.
Homomorphic encryption opens up new possibilities for secure outsourcing computation in CPS.


## Encrypted control

Encrypted control is an emerging research paradigm in the interdisciplinary field of control theory and cryptography that applies homomorphic encryption to decision making in control systems.
It exploits the ability of homomorphic encryption to compute encrypted feedback actions in controllers with encrypted controller parameters and encrypted sensor measurements.
This makes it possible to improve security against adversaries not only on communication channels but also on untrusted computers.
Moreover, encrypted controllers do not need to store encryption and decryption keys to perform the computation, thereby reducing key management costs.


## Requirements

ECLib now supports [Python 3.12](https://www.python.org/) and depends on [NumPy](https://numpy.org/doc/stable/index.html).
The codes in [Examples](../examples/examples.rst) requires [Matplotlib](https://matplotlib.org/) and [python-control](https://python-control.readthedocs.io/en/0.10.0/).


## License

[MIT license](../license.rst).