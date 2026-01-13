# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 21:34:58 2022

@author: James Joseph
"""

import cplot
import numpy as np

def f(z):
    return np.sin(z ** 3) / z


plt = cplot.plot(
    f,
    (-2.0, +2.0, 400),
    (-2.0, +2.0, 400),
    # abs_scaling=lambda x: x / (x + 1),  # how to scale the lightness in domain coloring
    # contours_abs=2.0,
    # contours_arg=(-np.pi / 2, 0, np.pi / 2, np.pi),
    # emphasize_abs_contour_1: bool = True,
    # colorspace: str = "cam16",
    # add_colorbars: bool = True,
    # add_axes_labels: bool = True,
    # saturation_adjustment: float = 1.28,
    # min_contour_length = None,
)
plt.show()

#%%


plt = cplot.plot(np.sin, (-5.0, +5.0, 400), (-5.0, +5.0, 400))
plt.show()