#! /usr/bin/env python3

"""Prime number utilities.

This module provides utility functions for generating prime numbers and semiprime
factors. The module includes functions for checking if an integer is prime, generating
a prime number with a specified bit length, generating a Sophie Germain prime and its
corresponding safe prime, and generating a pair of semiprime factors of the same bit
length.

Functions
---------
- is_prime
- get_prime
- get_safe_prime
- get_semiprime_factors
"""

from math import gcd

import eclib.randutils as ru


def is_prime(n: int, k: int = 50) -> bool:
    """
    Check if an integer `n` is prime.

    Parameters
    ----------
    n : int
        Integer to be checked for primality.
    k : int, optional
        The number of iterations for the Miller-Rabin primality test.

    Returns
    -------
    bool
        True if `n` is a prime number, False otherwise.

    Note
    ----
    The function uses the Miller-Rabin primality test to check if `n` is a prime
    number. The test is probabilistic and has a probability of failure less than
    `4^(-k)`. The parameter `k` determines the accuracy of the test.
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

    Parameters
    ----------
    bit_length : int
        Desired bit length of the prime number.

    Returns
    -------
    int
        Generated prime number.
    """

    p = ru.get_rand_bits(bit_length)
    while is_prime(p) is False:
        p = ru.get_rand_bits(bit_length)

    return p


def get_safe_prime(bit_length: int) -> tuple[int, int]:
    """
    Generates a Sophie Germain prime and its corresponding safe prime.

    Parameters
    ----------
    bit_length : int
        Desired bit length of the Sophie Germain prime.

    Returns
    -------
    sophie_germain_prime : int
        Sophie Germain prime.
    safe_prime : int
        Corresponding safe prime.
    """

    sophie_germain_prime = get_prime(bit_length)
    while is_prime(safe_prime := 2 * sophie_germain_prime + 1) is False:
        sophie_germain_prime = get_prime(bit_length)

    return sophie_germain_prime, safe_prime


def get_semiprime_factors(bit_length: int) -> tuple[int, int]:
    """
    Generates a pair of semiprime factors of the same bit length.

    Parameters
    ----------
    bit_length : int
        Desired bit length of the semiprime factors.

    Returns
    -------
    p : int
        First semiprime factor.
    q : int
        Second semiprime factor.
    """

    p = get_prime(bit_length)
    q = get_prime(bit_length)
    while gcd(p * q, (p - 1) * (q - 1)) != 1 or p == q:
        p = get_prime(bit_length)
        q = get_prime(bit_length)

    return p, q
