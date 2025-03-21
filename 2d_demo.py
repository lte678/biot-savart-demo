# This is a script to calculate the field in a 2D polygonal conductor with arbitrary polygonal cross section.
# It is exactly accurate and analytical.
# It is programmed according to the algorithm described in

# The evaluation of the Biot–Savart integral, J.-C. Suh. 2000
# Journal of Engineering Mathematics
# https://doi.org/10.1023/A:1004666000020

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///

from math import log, atan, isclose, pi
from itertools import pairwise
import numpy as np
import matplotlib.pyplot as plt


def I(ell, x, y, omega):
    return 0.5 * omega * (ell + I_1(ell, x, y))


def I_1(ell, x, y):
    r_0 = (x**2 + y**2)**0.5
    r_1 = ((x - ell)**2 + y**2)**0.5
    theta_0 = atan(abs(y) * ell / (r_0**2 - ell*x))
    if theta_0 < 0.0:
        theta_0 += pi
    assert theta_0 >= 0.0
    assert theta_0 <= pi
    return (ell - x)*log(r_1**2) + x*log(r_0**2) - 2*ell + 2*abs(y)*theta_0


def field_for_polygon(x: list, y: list, x_p, y_p, omega: float):
    """ (x_p, y_p) is the field point to evaluate at. """
    line_integral_x = 0.0
    line_integral_y = 0.0
    for (x_0, x_1), (y_0, y_1) in zip(pairwise(x), pairwise(y)):
        # Length of line segment
        ell = ((x_1 - x_0)**2 + (y_1 - y_0)**2)**0.5
        s_x = (x_1 - x_0) / ell
        s_y = (y_1 - y_0) / ell
        n_x = -s_y
        n_y = s_x
        assert isclose(s_x**2 + s_y**2, 1.0)
        assert isclose(n_x**2 + n_y**2, 1.0)

        x_p_loc = -(x_0 - x_p) * s_x - (y_0 - y_p) * s_y
        y_p_loc =  (x_0 - x_p) * n_x + (y_0 - y_p) * n_y
        # print(f"Evaluating for line ({x_0:.1f}, {y_0:.1f}) to ({x_1:.1f}, {y_1:.1f}) for field point ({x_p_loc:.1f}, {y_p_loc:.1f}) [local]")
        I_mag = I(ell, x_p_loc, y_p_loc, omega)
        line_integral_x += I_mag * n_x
        line_integral_y += I_mag * n_y
    
    return line_integral_y / (2 * pi), -line_integral_x / (2 * pi)


SIZE_X = 0.5
SIZE_Y = 0.1
poly_x = [-SIZE_X, SIZE_X, SIZE_X, -SIZE_X, -SIZE_X]
poly_y = [-SIZE_Y, -SIZE_Y, SIZE_Y,  SIZE_Y, -SIZE_Y]

X = np.linspace(-2.5, 2.5, 200)
Y = np.linspace(-2.5, 2.5, 200)
x_grid, y_grid = np.meshgrid(X, Y)

CURRENT_DENSITY = 2.0

np_field = np.vectorize(lambda x_p, y_p: field_for_polygon(poly_x, poly_y, x_p, y_p, CURRENT_DENSITY), signature='(),()->(),()')
B_x, B_y = np_field(x_grid, y_grid)

plt.rcParams["figure.figsize"] = (12, 10)
plt.streamplot(x_grid, y_grid, B_x, B_y)
plt.contourf(x_grid, y_grid, (B_x**2 + B_y**2)**0.5)
plt.plot(poly_x, poly_y, color='black', linewidth=3)
plt.colorbar()
plt.show()