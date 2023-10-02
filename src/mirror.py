import numpy as np
from numba.experimental import jitclass
import numba


class Mirror:
    distance: np.float64

    def __init__(self):
        print("Init mirror")
        self.distance = np.float64(1.0)

    def normal_and_height(self, uv: np.array):
        n = np.array([0.0, 0.0, 1.0])
        h = 0.0
        return n, h

    def brdf(self, light_dir, view_dir, normal):
        # Test
        return np.dot(light_dir, normal) * 0.5
