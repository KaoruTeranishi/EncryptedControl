#! /usr/bin/env python3


def is_generator(g: int, q: int, p: int) -> bool:
    if g <= 1 or g >= p:
        return False
    elif pow(g, q, p) == 1:
        return True
    else:
        return False


def is_element(m: int, q: int, p: int) -> bool:
    if m <= 0 or m >= p:
        return False
    elif pow(m, q, p) == 1:
        return True
    else:
        return False


def get_generator(q: int, p: int) -> int:
    g = 2
    while is_generator(g, q, p) is False:
        g += 1

    return g
