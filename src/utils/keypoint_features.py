import numpy as np

from src.config.geometry import ANGLE_TRIPLETS, KEYPOINT_MAP, POINT_CONNECTIONS
from src.utils.geometry import calculate_angle, calculate_distance, slug, triangle_area


def build_xgb_feature_dict(keypoints: np.ndarray) -> dict[str, float]:
    required_kpts = len(KEYPOINT_MAP)
    if keypoints.shape[0] < required_kpts:
        raise ValueError(f"Esperava {required_kpts} pontos-chave, mas recebeu {keypoints.shape[0]}")

    keypoints = keypoints[:required_kpts]
    keypoint_map = {KEYPOINT_MAP[i]: keypoints[i] for i in range(required_kpts)}

    features: dict[str, float] = {}

    for p1, p2 in POINT_CONNECTIONS:
        features[f"dist_{slug(p1)}__{slug(p2)}"] = calculate_distance(keypoint_map[p1], keypoint_map[p2])

    for a, b, c in ANGLE_TRIPLETS:
        area = triangle_area(keypoint_map[a], keypoint_map[b], keypoint_map[c])
        if area <= 1e-12:
            continue

        features[f"angle_{slug(a)}__{slug(b)}__{slug(c)}_at_{slug(a)}"] = calculate_angle(
            keypoint_map[b], keypoint_map[a], keypoint_map[c]
        )
        features[f"angle_{slug(a)}__{slug(b)}__{slug(c)}_at_{slug(b)}"] = calculate_angle(
            keypoint_map[a], keypoint_map[b], keypoint_map[c]
        )
        features[f"angle_{slug(a)}__{slug(b)}__{slug(c)}_at_{slug(c)}"] = calculate_angle(
            keypoint_map[a], keypoint_map[c], keypoint_map[b]
        )
        features[f"triangle_area_{slug(a)}__{slug(b)}__{slug(c)}"] = area

    return features
