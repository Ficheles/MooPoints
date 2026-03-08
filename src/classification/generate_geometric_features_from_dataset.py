import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.classification.inference_pipeline import ANGLE_TRIPLETS, DISTANCE_PAIRS, KEYPOINT_MAP, calculate_angle, calculate_distance

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gera features geométricas para classificação a partir de imagens do dataset_classification "
            "usando modelo YOLO de keypoints."
        )
    )
    parser.add_argument("--dataset-root", default="dataset_classification", help="Diretório raiz do dataset de classificação.")
    parser.add_argument("--fold", type=int, default=0, help="Fold a utilizar para train/val (padrão: 0).")
    parser.add_argument("--model-path", default="models/yolo11x-pose.pt", help="Modelo YOLO pose para extração de keypoints.")
    parser.add_argument(
        "--output-csv",
        default="dataset_classification/geometric_features.csv",
        help="CSV de saída com as features geométricas.",
    )
    return parser.parse_args()


def find_images(split_root: Path):
    items = []
    if not split_root.exists():
        return items

    for class_dir in sorted([d for d in split_root.iterdir() if d.is_dir()]):
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((path, class_dir.name))
    return items


def extract_keypoints(model: YOLO, image_path: Path) -> np.ndarray:
    conf_schedule = (0.25, 0.15, 0.08, 0.05)
    imgsz_schedule = (640, 960, 1280)

    for imgsz in imgsz_schedule:
        for conf in conf_schedule:
            results = model.predict(source=str(image_path), task="pose", conf=conf, imgsz=imgsz, verbose=False)
            if not results:
                continue
            keypoints_obj = getattr(results[0], "keypoints", None)
            if keypoints_obj is None or keypoints_obj.xy is None:
                continue

            keypoints_batch = keypoints_obj.xy.cpu().numpy()
            if len(keypoints_batch) == 0:
                continue

            keypoints = keypoints_batch[0]
            if keypoints.shape[0] >= len(KEYPOINT_MAP):
                return keypoints[: len(KEYPOINT_MAP)]

    raise ValueError("no_keypoints_detected")


def keypoints_to_features(keypoints: np.ndarray) -> dict[str, float]:
    keypoint_map = {KEYPOINT_MAP[i]: keypoints[i] for i in range(len(KEYPOINT_MAP))}

    features = {}
    for p1, p2 in DISTANCE_PAIRS:
        features[f"dist_{p1}_{p2}"] = float(calculate_distance(keypoint_map[p1], keypoint_map[p2]))

    for p1, p2, p3 in ANGLE_TRIPLETS:
        features[f"angle_{p1}_{p2}_{p3}"] = float(calculate_angle(keypoint_map[p1], keypoint_map[p2], keypoint_map[p3]))

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

    model = YOLO(str(model_path))

    fold_root = dataset_root / f"fold_{args.fold}"
    train_items = find_images(fold_root / "train")
    val_items = find_images(fold_root / "val")
    test_items = find_images(dataset_root / "test")

    if not train_items and not val_items and not test_items:
        raise FileNotFoundError("Nenhuma imagem encontrada em train/val/test para extração de features.")

    rows = []
    failures = []

    for split_name, items in (("train", train_items), ("val", val_items), ("test", test_items)):
        for image_path, class_name in items:
            try:
                keypoints = extract_keypoints(model, image_path)
                features = keypoints_to_features(keypoints)
                rows.append(
                    {
                        "image_path": str(image_path.resolve()),
                        "class_name": class_name,
                        "split": split_name,
                        **features,
                    }
                )
            except ValueError as exc:
                failures.append((str(image_path), str(exc)))

    if not rows:
        raise RuntimeError("Nenhuma feature válida foi gerada.")

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("=" * 70)
    print("Geração de features geométricas concluída")
    print(f"- output_csv: {output_csv}")
    print(f"- linhas geradas: {len(df)}")
    print(f"- falhas: {len(failures)}")
    if failures:
        print("- exemplo de falha:")
        sample = failures[0]
        print(f"  - {sample[0]} :: {sample[1]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
