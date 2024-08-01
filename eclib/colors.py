#! /usr/bin/env python3

"""
colors.py

This module defines universal colors in RGB format. The colors are used for plotting
graphs and figures in a consistent manner.

Classes:
    Colors: Defines universal colors in RGB format.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Colors:
    """
    A class that defines universal colors in RGB format.

    Attributes:
        red (tuple): The RGB values for the color red.
        yellow (tuple): The RGB values for the color yellow.
        green (tuple): The RGB values for the color green.
        blue (tuple): The RGB values for the color blue.
        sky_blue (tuple): The RGB values for the color sky blue.
        pink (tuple): The RGB values for the color pink.
        orange (tuple): The RGB values for the color orange.
        purple (tuple): The RGB values for the color purple.
        brown (tuple): The RGB values for the color brown.
        light_pink (tuple): The RGB values for the color light pink.
        cream (tuple): The RGB values for the color cream.
        light_yellowgreen (tuple): The RGB values for the color light yellowgreen.
        light_sky_blue (tuple): The RGB values for the color light sky blue.
        beige (tuple): The RGB values for the color beige.
        light_green (tuple): The RGB values for the color light green.
        light_purple (tuple): The RGB values for the color light purple.
        light_gray (tuple): The RGB values for the color light gray.
        gray (tuple): The RGB values for the color gray.
        white (tuple): The RGB values for the color white.
        black (tuple): The RGB values for the color black.

    References:
        https://jfly.uni-koeln.de/colorset/
    """

    # accent colors
    red = (1.0, 75 / 255, 0.0)
    yellow = (1.0, 241 / 255, 0.0)
    green = (3 / 255, 175 / 255, 122 / 255)
    blue = (0.0, 90 / 255, 1.0)
    sky_blue = (77 / 255, 196 / 255, 1.0)
    pink = (1.0, 128 / 255, 130 / 255)
    orange = (246 / 255, 170 / 255, 0.0)
    purple = (153 / 255, 0.0, 153 / 255)
    brown = (128 / 255, 64 / 255, 0.0)

    # base colors
    light_pink = (1.0, 202 / 255, 191 / 255)
    cream = (1.0, 1.0, 128)
    light_yellowgreen = (216 / 255, 242 / 255, 85 / 255)
    light_sky_blue = (191 / 255, 228 / 255, 1.0)
    beige = (1.0, 202 / 255, 128 / 255)
    light_green = (119 / 255, 217 / 255, 168 / 255)
    light_purple = (201 / 255, 172 / 255, 230)

    # neutral colors
    light_gray = (200 / 255, 200 / 255, 203 / 255)
    gray = (132 / 255, 145 / 255, 158 / 255)
    white = (1.0, 1.0, 1.0)
    black = (0.0, 0.0, 0.0)
