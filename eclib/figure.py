#! /usr/bin/env python3

"""Figure settings and universal colors.

This module provides a dataclass for universal colors in RGB format and a function for
default figure settings for matplotlib.

Classes
-------
- Colors

Functions
---------
- setup
"""

from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class Colors:
    """
    Represents universal colors in RGB format.

    Attributes
    ----------
    red : tuple
    yellow : tuple
    green : tuple
    blue : tuple
    sky_blue : tuple
    pink : tuple
    orange : tuple
    purple : tuple
    brown : tuple
    light_pink : tuple
    cream : tuple
    light_yellowgreen : tuple
    light_sky_blue : tuple
    beige : tuple
    light_green : tuple
    light_purple : tuple
    light_gray : tuple
    gray : tuple
    white : tuple
    black : tuple

    References
    ----------
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


def setup() -> None:
    """
    Sets up the default figure settings for matplotlib.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # size
    plt.rcParams["figure.figsize"] = (1.62 * 2, 1 * 2)

    # font
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10

    # axis
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.major.width"] = 1.0
    plt.rcParams["ytick.major.width"] = 1.0

    # legend
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.framealpha"] = 1.0
    plt.rcParams["legend.edgecolor"] = "#000000"
    plt.rcParams["legend.handlelength"] = 1.0

    # grid
    plt.rcParams["axes.grid"] = False
