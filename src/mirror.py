import numpy as np
import cupy as cp
from pathlib import Path
import os
import src.math_utils as mu

class Mirror:
    __resolution: list[int]
    __size: list[float]
    __delta_xy: list[float]
    __distance: float
    __kernel_grid_size: int
    __kernel_block_size: int
    __heightmap_kernel: cp.RawKernel
    __canvas_kernel: cp.RawKernel
    __heightmap: cp.ndarray
    __canvas: cp.ndarray

    def __init__(self, resolution: list[int], size: list[float], distance: float):
        self.__resolution = resolution
        self.__size = size
        self.__delta_xy = [self.__size[0] / self.__resolution[0], self.__size[1] / self.__resolution[1]]
        self.__distance = distance
        self.__kernel_grid_size, self.__kernel_block_size = mu.get_grid_size_block_size(self.__resolution, False)
        self._init_kernels()

    def _init_kernels(self):
        heightmap_kernel_source = Path("src/cuda_kernels/heightmap.cu").read_text()
        heightmap_func_name = 'heightmap'
        self.__heightmap_kernel = cp.RawModule(
            code=heightmap_kernel_source,
            name_expressions=[heightmap_func_name],
            options=("-std=c++20", f"-I{os.path.abspath('src')}")
        ).get_function(heightmap_func_name)

        canvas_kernel_source = Path("src/cuda_kernels/canvas.cu").read_text()
        canvas_func_name = 'canvas'
        self.__canvas_kernel = cp.RawModule(
            code=canvas_kernel_source,
            name_expressions=[canvas_func_name],
            options=("-std=c++20", f"-I{os.path.abspath('src')}"),
        ).get_function(canvas_func_name)


    def render_heightmap(self) -> np.ndarray:
        """
        Render the heightmap of the mirror.

        :return: heightmap as a Numpy ndarray
        """
        self.__heightmap = cp.zeros(self.__resolution, dtype=cp.float64)
        self.__heightmap_kernel(
            self.__kernel_grid_size,
            self.__kernel_block_size,
            (
                self.__heightmap,
                cp.float64(self.__delta_xy[0]),
                cp.float64(self.__delta_xy[1]),
            )
        )
        return cp.asnumpy(self.__heightmap)

    def render_canvas(self) -> np.ndarray:
        """
        Render a canvas image reflected by the mirror.

        :return: canvas image as a Numpy ndarray
        """
        if self.__heightmap is None:
            raise RuntimeError("Rendering canvas image is impossible without heightmap. Create a heightmap first!")
        self.__canvas = cp.zeros(self.__resolution, dtype=cp.float64)
        self.__canvas_kernel(
            self.__kernel_grid_size,
            self.__kernel_block_size,
            (
                self.__heightmap,
                self.__canvas,
                cp.float64(self.__delta_xy[0]),
                cp.float64(self.__delta_xy[1]),
                self.__distance,
            )
        )
        return cp.asnumpy(self.__canvas)
