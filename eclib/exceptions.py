#! /usr/bin/env python3


class EncodingError(Exception):
    pass


class DecodingError(Exception):
    pass


class EncryptionError(Exception):
    pass


class DecryptionError(Exception):
    pass


class HomomorphicOperationError(Exception):
    pass


class CiphertextUpdateError(Exception):
    pass
