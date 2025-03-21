# This is a script to calculate the field in an arbitrary polyhedron.
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

from itertools import pairwise
from math import pi, log, asin, sqrt, isclose
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_polygon_normal(polygon):
    """ Each row is a point on the polygon. Must be closed by including the first point at the end. """
    if polygon.shape[0] < 3:
        raise ValueError
    
    r0 = polygon[0]
    norm = np.cross(polygon[1] - r0, polygon[2] - r0)
    if polygon.shape[0] > 3:
        # A triangle is always coplanar
        for poly_vertex in polygon[3:]:
            if not isclose(np.dot(norm, poly_vertex - r0), 0.0):
                raise ValueError('Polygon is not planar')
    return norm / np.linalg.norm(norm)


def edge_contribution(e, r_0, r_1, x, y, ell):
    # This guy is scalar
    E = log((r_1 + ell - x) / (r_0 - x))
    F = ((y**2 + e*r_0) / (r_0 + e))**2 + ((y**2 + e*r_1) / (r_1 + e))**2 - y**2
    H = (sqrt(y**2 - e**2) * (y**2*ell + e*(ell - x)*r_0 + e*x*r_1)) / (y**2*(r_0 + e)*(r_1 + e))
    if F > 0:
        beta = asin(H)
    else:
        beta = pi - asin(H)
    
    return E - beta * e / sqrt(y**2 - e**2)


def L(polygon, r, omega):
    n = get_polygon_normal(polygon)
    assert isclose(np.linalg.norm(n), 1.0)
    e_n = n
    if np.dot(e_n, r) < 0.0:
        e_n = -e_n
    
    gamma = -np.cross(n, omega)

    K_1_sum = 0.0
    for vertex_0, vertex_1 in pairwise(polygon):
        ell = np.linalg.norm(vertex_1 - vertex_0)
        s_i = (vertex_1 - vertex_0) / ell
        assert isclose(np.linalg.norm(s_i), 1.0)
        r_0 = vertex_0 - r
        r_1 = vertex_1 - r
        e = np.dot(e_n, r_0)
        k = np.cross(s_i, r_0)
        k /= np.linalg.norm(k)
        x = -np.dot(r_0, s_i)
        y =  np.dot(r_0, np.cross(s_i, k))
        # print(f'r_0={r_0}, y={y}, e={e}')
        assert abs(e) <= abs(y)
        # It is just a coordinate transformation it should still have the same length
        assert isclose(sqrt(x**2 + y**2), np.linalg.norm(r_0))
        K_1 = edge_contribution(e, np.linalg.norm(r_0), np.linalg.norm(r_1), x, y, ell)
        b_i = np.dot(np.cross(n, r), s_i)
        K_1_sum += b_i * K_1
    return gamma * K_1
    
    
def field_for_volume(polygons, r, omega):
    field = np.zeros(3)
    for polygon in polygons:
        field += L(polygon, r, omega)
    return -field / (4*pi)


vertices = np.array([
    [-1.0, -1.0, -1.0],
    [1.0,  -1.0, -1.0],
    [1.0,   1.0, -1.0],
    [-1.0,  1.0, -1.0],
    
    [-1.0, -1.0, 1.0],
    [1.0,  -1.0, 1.0],
    [1.0,   1.0, 1.0],
    [-1.0,  1.0, 1.0],
])

CURRENT_DENSITY = np.array([0.0, 0.0, 2.0])
face1 = np.array([vertices[0], vertices[1], vertices[2], vertices[3], vertices[0]])
face2 = np.array([vertices[4], vertices[5], vertices[6], vertices[7], vertices[4]])
face3 = np.array([vertices[0], vertices[1], vertices[5], vertices[4], vertices[0]])
face4 = np.array([vertices[2], vertices[3], vertices[7], vertices[6], vertices[2]])
face5 = np.array([vertices[1], vertices[2], vertices[6], vertices[5], vertices[1]])
face6 = np.array([vertices[0], vertices[3], vertices[7], vertices[4], vertices[0]])
box = [face1, face2, face3, face4, face5, face6]

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.add_collection3d(Poly3DCollection(box, alpha=0.2))
#plt.show()

r = np.array([-5.0, 0.0, 5.0])
print(field_for_volume(box, r, CURRENT_DENSITY))


## Plotting
X = np.linspace(-2.5, 2.5, 50)
Y = np.linspace(-2.5, 2.5, 50)
x_grid, y_grid = np.meshgrid(X, Y)

np_field = np.vectorize(lambda x_p, y_p: field_for_volume(box, np.array([x_p, y_p, 0.0]), CURRENT_DENSITY), signature='(),()->(n)')
B = np_field(x_grid, y_grid)
B_x = B[:, :, 0]
B_y = B[:, :, 1]

plt.rcParams["figure.figsize"] = (12, 10)
plt.streamplot(x_grid, y_grid, B_x, B_y)
plt.contourf(x_grid, y_grid, (B_x**2 + B_y**2)**0.5)
plt.colorbar()
plt.show()