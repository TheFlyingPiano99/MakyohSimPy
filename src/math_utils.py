import numpy as np
import math


def length(vec):
    return math.sqrt(np.dot(vec, vec))


def normalize(vec):
    m = 1.0 / math.sqrt(np.dot(vec, vec))
    return np.multiply(vec, np.array([m, m, m]))
