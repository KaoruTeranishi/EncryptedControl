#! /usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# size
plt.rcParams['figure.figsize'] = (1.62 * 2, 1 * 2)
# font
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
# axis
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
# legend
plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['legend.edgecolor'] = '#000000'
plt.rcParams['legend.handlelength'] = 1.0
# grid
plt.rcParams['axes.grid'] = False
