# Overview

ECLib provides an intuitive and unified interface for accessing the functions of various homomorphic encryption schemes.
Furthermore, it provides a set of tools for imprementing encrypted control systems.
This document presents a detailed explanation of the fundamental concepts and usage of ECLib.

To run the codes in this document, you need to install the ECLib package and import one of the encryption scheme modules.
For example, to import the ElGamal encryption scheme, you can use the following code.

```python
>>> import eclib.elgamal
```

Each encryption scheme module includes functions for key generation, encryption, decryption, homomorphic operation(s), encoding, and decoding.

In addition, the system description module contains functions for describing a plant, sensor, actuator, (encrypted) controller, and system operator in (encrypted) control systems.
You can import the system description module as follows.

```python
>>> import eclib.system
```

This module helps you to implement (encrypted) control systems in a simple and intuitive way.