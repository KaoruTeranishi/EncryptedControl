#! /usr/bin/env python3

from math import gcd

import eclib.randutils as ru


# Miller–Rabin primality test
def is_prime(n: int, k: int = 50) -> bool:
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
    p = ru.get_rand_bits(bit_length)
    while is_prime(p) is False:
        p = ru.get_rand_bits(bit_length)

    return p


def get_safe_prime(bit_length: int) -> tuple[int, int]:
    p = get_prime(bit_length)
    while is_prime(2 * p + 1) is False:
        p = get_prime(bit_length)

    return p, 2 * p + 1


def get_semiprime_factors(bit_length: int) -> tuple[int, int]:
    p = get_prime(bit_length)
    q = get_prime(bit_length)
    while gcd(p * q, (p - 1) * (q - 1)) != 1 or p == q:
        p = get_prime(bit_length)
        q = get_prime(bit_length)

    return p, q
