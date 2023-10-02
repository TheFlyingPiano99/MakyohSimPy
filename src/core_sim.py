from math import acos
from types import resolve_bases
import numpy as np
from numba import jit
import random
from src.mirror import Mirror
import src.math_utils as mu
from joblib import Parallel, delayed


def calculate_irradiance_at_pdoint(canvas_pos: np.array, mirror: Mirror):
    M = 0.0
    canvas_normal = np.array([0.0, 0.0, -1.0])
    light_dir = np.array([0.0, 0.0, 1.0])
    light_power_density = 10.0
    mirror_resolution = 50
    for x in range(0, mirror_resolution):
        for y in range(0, mirror_resolution):
            mirror_u = x / float(mirror_resolution)
            mirror_v = y / float(mirror_resolution)
            # Apply perturbation to reduce sampling artifacts:
            mirror_u += random.randint(-1000, 1000) / 4000.0
            mirror_v += random.randint(-1000, 1000) / 4000.0
            # Skip position if outside of mirror surface
            if mirror_u > -2.0 or mirror_u < 0.0 or mirror_v > 1.0 or mirror_v < 0.0:
                continue
            mirror_normal, mirror_height = mirror.normal_and_height(
                np.array([mirror_u, mirror_v])
            )
            mirror_pos = np.array([mirror_u, mirror_v, mirror.distance])
            to_mirror = mu.normalize(mirror_pos - canvas_pos)
            light_distance_squared = 1.0  # Directional light (No attenuation)
            mirror_delta = 1.0 / float(mirror_resolution)

            # Calculating correction term:
            a_x = mu.length(
                np.array([mirror_pos[0], mirror_pos[1], -mirror.distance])
                - np.array([mirror_delta, 0.0, 0.0]) * 0.5
                - canvas_pos
            )
            b_x = mu.length(
                np.array([mirror_pos[0], mirror_pos[1], -mirror.distance])
                + np.array([mirror_delta, 0.0, 0.0]) * 0.5
                - canvas_pos
            )
            c_x = mirror_delta
            delta_x = acos((a_x * a_x + b_x * b_x - c_x * c_x) / (2.0 * a_x * b_x))

            a_y = mu.length(
                np.array([mirror_pos[0], mirror_pos[1], mirror.distance])
                - np.array([0.0, mirror_delta, 0.0]) * 0.5
                - canvas_pos
            )
            b_y = mu.length(
                np.array([mirror_pos[0], mirror_pos[1], mirror.distance])
                - np.array([0.0, mirror_delta, 0.0]) * 0.5
                - canvas_pos
            )
            c_y = mirror_delta
            delta_y = acos((a_y * a_y + b_y * b_y - c_y * c_y) / (2.0 * a_y * b_y))
            omega = delta_x * delta_y

            # Calculate radiance at mirror surface:
            L_in = (
                light_power_density
                / light_distance_squared
                * max(np.dot(light_dir, mirror_normal), 0.0)
            ) * mirror.brdf(light_dir, -to_mirror, mirror_normal)
            M += L_in * omega * np.dot(to_mirror, canvas_normal)

    return M


def render_reflection(resolution: int, mirror: Mirror):
    def func(x:int, y:int):
        print("*")
        u = x / float(resolution)
        v = y / float(resolution)
        canvas_pos = np.array([u, v, 0.0])
        return calculate_irradiance_at_point(canvas_pos, mirror)

    return np.fromfunction(function=func, shape=(resolution, resolution), dtype=object)


def render_height_map(resolution: int):
    height_map = np.zeros(shape=(1024, 1024), dtype=np.float_)

    return height_map
