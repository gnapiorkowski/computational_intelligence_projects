import numpy as np
import math


def vec_angle(v1, v2):
    vec_cos = (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    vec_rad = np.arccos(vec_cos)
    return np.rad2deg(vec_rad)


x1 = np.array([1, 2, 3])
x2 = np.array([-2, -1, 5])
print(vec_angle(x1, x2))
