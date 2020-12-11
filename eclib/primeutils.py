#! /usr/bin/env python

from eclib.randutils import *

# Miller–Rabin primality test
def is_prime(n, k=50):
    if n == 2:
        return True
    elif n < 2 or n % 2 == 0:
        return False
    else:
        d = n - 1
        while d % 2 == 0:
            d >>= 1
        
        for i in range(k):
            a = get_rand(1, n)
            t = d
            y = pow(a, t, n)
            while t != n - 1 and y != 1 and y != n - 1: 
                y = pow(y, 2, n)
                t <<= 1
            if y != n - 1 and t % 2 == 0:
                return False
        
        return True

def get_prime(bit_length):
    p = get_rand_bits(bit_length)
    while not is_prime(p):
        p = get_rand_bits(bit_length)
    
    return p

def get_safe_prime(bit_length):
    p = get_prime(bit_length)
    while not is_prime(2 * p + 1):
        p = get_prime(bit_length)
    
    return p, 2 * p + 1
