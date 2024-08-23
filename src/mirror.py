import numpy as np
import cupy as cp

class Mirror:
    __resolution: list[int]
    __distance: float

    def __init__(self, resolution: list[int], distance: float):
        self.__resolution = resolution
        self.__distance = distance

    def render_canvas(self) -> np.ndarray:
        """
        Render a canvas image reflected by the mirror.
        """
        canvas = cp.zeros(self.__resolution, dtype=cp.float64)
        return cp.asnumpy(canvas)

    def render_heightmap(self) -> np.ndarray:
        heightmap = cp.zeros(self.__resolution, dtype=cp.float64)
        return cp.asnumpy(heightmap)