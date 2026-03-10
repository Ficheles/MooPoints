import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

KEYPOINT_MAP = {
    0: "withers",
    1: "back",
    2: "hook up",
    3: "hook down",
    4: "hip",
    5: "tail head",
    6: "pin up",
    7: "pin down",
}

POINT_CONNECTIONS = [
    ("withers", "back"),
    ("withers", "hook up"),
    ("withers", "hook down"),
    ("back", "hip"),
    ("back", "hook up"),
    ("back", "hook down"),
    ("hook up", "hook down"),
    ("hook up", "hip"),
    ("hook down", "hip"),
    ("hip", "tail head"),
    ("hook up", "tail head"),
    ("hook down", "tail head"),
    ("hook up", "pin up"),
    ("hook down", "pin down"),
    ("tail head", "pin up"),
    ("tail head", "pin down"),
    ("pin up", "pin down"),
]

ANGLE_TRIPLETS = [
    ("withers", "back", "hook up"),
    ("withers", "back", "hook down"),
    ("withers", "hook up", "hook down"),
    ("back", "hook up", "hook down"),
    ("back", "hook up", "hip"),
    ("back", "hook down", "hip"),
    ("hook up", "hook down", "hip"),
    ("hook up", "hook down", "tail head"),
    ("hook up", "tail head", "pin up"),
    ("hook down", "tail head", "pin down"),
    ("tail head", "pin up", "pin down"),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Roda predição YOLO pose em data/datasets/classifications, salva payloads em labels e "
            "extrai features geométricas (distâncias + ângulos de triângulos) para CSV."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="data/datasets/classifications",
        help="Raiz do dataset de classificação com fold_*/train/images, fold_*/val/images e test/images.",
    )
    parser.add_argument(
        "--model-path",
        default="models/yolo/best.pt",
        help="Modelo YOLO Pose para predição de keypoints.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/datasets/classifications/geometric_features.csv",
        help="CSV consolidado de features geométricas.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confiança mínima para predição de pose.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Tamanho de inferência para predição de pose.",
    )
    return parser.parse_args()


def slug(name: str) -> str:
    return name.replace(" ", "_")


def extract_cow_id_from_filename(image_path: Path) -> str:
    stem = image_path.stem
    tokens = stem.split("_")
    if tokens and tokens[0]:
        return tokens[0]
    return stem


def discover_splits(dataset_root: Path):
    split_specs = []

    for fold_dir in sorted(dataset_root.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        for split in ("train", "val"):
            split_root = fold_dir / split
            images_dir = split_root / "images" if (split_root / "images").exists() else split_root
            labels_dir = fold_dir / split / "labels"
            if images_dir.exists():
                split_specs.append((fold_dir.name, split, images_dir, labels_dir))

    test_root = dataset_root / "test"
    test_images = test_root / "images" if (test_root / "images").exists() else test_root
    test_labels = dataset_root / "test" / "labels"
    if test_images.exists():
        split_specs.append(("test", "test", test_images, test_labels))

    return split_specs


def list_class_images(images_root: Path):
    samples = []
    class_dirs = sorted([d for d in images_root.iterdir() if d.is_dir()])
    for class_dir in class_dirs:
        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, class_dir.name))
    return samples


def serialize_result_payload(result) -> dict:
    payload = {
        "boxes": None,
        "keypoints_xy": None,
        "keypoints_conf": None,
    }

    boxes = getattr(result, "boxes", None)
    if boxes is not None and boxes.xyxy is not None:
        payload["boxes"] = boxes.xyxy.cpu().numpy().tolist()

    keypoints = getattr(result, "keypoints", None)
    if keypoints is not None and keypoints.xy is not None:
        payload["keypoints_xy"] = keypoints.xy.cpu().numpy().tolist()

    if keypoints is not None and keypoints.conf is not None:
        payload["keypoints_conf"] = keypoints.conf.cpu().numpy().tolist()

    return payload


def select_first_keypoints(result) -> np.ndarray:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.xy is None:
        raise ValueError("no_keypoints")

    points = keypoints.xy.cpu().numpy()
    if len(points) == 0:
        raise ValueError("no_keypoints")

    first = points[0]
    if first.shape[0] < len(KEYPOINT_MAP):
        raise ValueError("insufficient_keypoints")

    return first[: len(KEYPOINT_MAP)]


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    v1 = p1 - p2
    v2 = p3 - p2
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= 1e-12:
        return float("nan")

    cos_theta = float(np.dot(v1, v2) / denom)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(np.degrees(math.acos(cos_theta)))


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    return float(abs(np.cross(p2 - p1, p3 - p1)) / 2.0)


def build_features_from_keypoints(points: np.ndarray):
    kp = {KEYPOINT_MAP[i]: points[i] for i in range(len(KEYPOINT_MAP))}
    features = {}

    for p1, p2 in POINT_CONNECTIONS:
        features[f"dist_{slug(p1)}__{slug(p2)}"] = distance(kp[p1], kp[p2])

    for a, b, c in ANGLE_TRIPLETS:
        area = triangle_area(kp[a], kp[b], kp[c])
        if area <= 1e-12:
            continue

        features[f"angle_{slug(a)}__{slug(b)}__{slug(c)}_at_{slug(a)}"] = angle(kp[b], kp[a], kp[c])
        features[f"angle_{slug(a)}__{slug(b)}__{slug(c)}_at_{slug(b)}"] = angle(kp[a], kp[b], kp[c])
        features[f"angle_{slug(a)}__{slug(b)}__{slug(c)}_at_{slug(c)}"] = angle(kp[a], kp[c], kp[b])
        features[f"triangle_area_{slug(a)}__{slug(b)}__{slug(c)}"] = area

    return features


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    model_path = Path(args.model_path)
    output_csv = Path(args.output_csv)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root não encontrado: {dataset_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo YOLO pose não encontrado: {model_path}")

    split_specs = discover_splits(dataset_root)
    if not split_specs:
        raise FileNotFoundError("Nenhum split encontrado. Esperado fold_*/train|val/images e/ou test/images.")

    model = YOLO(str(model_path))
    rows = []
    failures = []
    payload_count = 0

    for fold_name, split_name, images_dir, labels_dir in split_specs:
        labels_dir.mkdir(parents=True, exist_ok=True)
        samples = list_class_images(images_dir)

        for image_path, class_name in samples:
            cow_id = extract_cow_id_from_filename(image_path)
            rel = image_path.relative_to(images_dir)
            payload_path = labels_dir / rel.with_suffix(".json")
            payload_path.parent.mkdir(parents=True, exist_ok=True)

            results = model.predict(
                source=str(image_path),
                task="pose",
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )

            if not results:
                failures.append({
                    "image_path": str(image_path),
                    "fold": fold_name,
                    "split": split_name,
                    "reason": "no_result",
                })
                continue

            result = results[0]
            payload = serialize_result_payload(result)
            payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            payload_count += 1

            try:
                points = select_first_keypoints(result)
                features = build_features_from_keypoints(points)
            except ValueError as exc:
                failures.append({
                    "image_path": str(image_path),
                    "fold": fold_name,
                    "split": split_name,
                    "reason": str(exc),
                })
                continue

            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "payload_path": str(payload_path.resolve()),
                    "fold": fold_name,
                    "split": split_name,
                    "class_name": class_name,
                    "cow_id": cow_id,
                    **features,
                }
            )

    if not rows:
        raise RuntimeError("Nenhuma feature foi extraída. Verifique modelo e imagens.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print("=" * 70)
    print("Extração geométrica concluída")
    print(f"- modelo: {model_path}")
    print(f"- payloads salvos: {payload_count}")
    print(f"- linhas no CSV: {len(df)}")
    print(f"- falhas: {len(failures)}")
    print(f"- output_csv: {output_csv}")
    if failures:
        sample = failures[0]
        print(f"- exemplo falha: {sample['image_path']} :: {sample['reason']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
