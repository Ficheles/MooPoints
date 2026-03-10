import math

import numpy as np


def slug(name: str) -> str:
    return name.replace(" ", "_")


def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    v1 = p1 - p2
    v2 = p3 - p2
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= 1e-12:
        return float("nan")

    cos_theta = float(np.dot(v1, v2) / denom)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(math.acos(cos_theta)))


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    return float(abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])) / 2.0)
