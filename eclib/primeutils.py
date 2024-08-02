#! /usr/bin/env python3

"""
primeutils.py

This module provides utility functions for generating prime numbers and semiprime
factors. The module includes functions for checking if an integer is prime, generating
a prime number with a specified bit length, generating a Sophie Germain prime and its
corresponding safe prime, and generating a pair of semiprime factors of the same bit
length.

Functions:
    is_prime: Check if an integer is prime.
    get_prime: Generate a prime number with the specified bit length.
    get_safe_prime: Generate a Sophie Germain prime and its corresponding safe prime.
    get_semiprime_factors: Generate a pair of semiprime factors of the same bit length.

Dependencies:
    eclib.randutils: Utility functions for generating random numbers.
"""

from math import gcd

import eclib.randutils as ru


def is_prime(n: int, k: int = 50) -> bool:
    """
    Check if an integer `n` is prime.

    Args:
        n (int): Integer to be checked for primality.
        k (int, default = 50): The number of iterations for the Miller-Rabin primality
            test.

    Returns:
        bool: True if `n` is a prime number, False otherwise.

    Note:
        The function uses the Miller-Rabin primality test to check if `n` is a prime
        number. The test is probabilistic and has a probability of failure less than
        4^(-k).
    """
    if n == 2:
        return True

    elif n < 2 or n % 2 == 0:
        return False

    else:
        d = n - 1
        while d % 2 == 0:
            d >>= 1

        for _ in range(k):
            a = ru.get_rand(1, n)
            t = d
            y = pow(a, t, n)
            while t != n - 1 and y != 1 and y != n - 1:
                y = pow(y, 2, n)
                t <<= 1

            if y != n - 1 and t % 2 == 0:
                return False

        return True


def get_prime(bit_length: int) -> int:
    """
        Generates a prime number with the specified bit length.

    Parameters:
        bit_length (int): Desired bit length of the prime number.

    Returns:
        int: The generated prime number.
    """

    p = ru.get_rand_bits(bit_length)
    while is_prime(p) is False:
        p = ru.get_rand_bits(bit_length)

    return p


def get_safe_prime(bit_length: int) -> tuple[int, int]:
    """
    Generates a Sophie Germain prime and its corresponding safe prime.

    Args:
        bit_length (int): Desired bit length of the Sophie Germain prime.

    Returns:
        tuple[int, int]: Tuple containing the Sophie Germain prime and its
            corresponding safe prime.
    """

    p = get_prime(bit_length)
    while is_prime(2 * p + 1) is False:
        p = get_prime(bit_length)

    return p, 2 * p + 1


def get_semiprime_factors(bit_length: int) -> tuple[int, int]:
    """
    Generates a pair of semiprime factors of the same bit length.

    Args:
        bit_length (int): Desired bit length of the semiprime factors.

    Returns:
        tuple[int, int]: Tuple containing the semiprime factors.
    """

    p = get_prime(bit_length)
    q = get_prime(bit_length)
    while gcd(p * q, (p - 1) * (q - 1)) != 1 or p == q:
        p = get_prime(bit_length)
        q = get_prime(bit_length)

    return p, q
