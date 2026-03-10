import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.classification.inference_pipeline import KEYPOINT_MAP, build_feature_dict

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
    parser.add_argument("--model-path", default="models/yolo/yolo11x-pose.pt", help="Modelo YOLO pose para extração de keypoints.")
    parser.add_argument(
        "--output-csv",
        default="dataset_classification/geometric_features.csv",
        help="CSV de saída com as features geométricas.",
    )
    parser.add_argument(
        "--output-report",
        default="dataset_classification/geometric_features_report.json",
        help="JSON de relatório do processo de extração.",
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
    return build_feature_dict(keypoints)


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    model_path = Path(args.model_path)
    output_csv = Path(args.output_csv)
    output_report = Path(args.output_report)

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
    split_totals = {
        "train": len(train_items),
        "val": len(val_items),
        "test": len(test_items),
    }
    split_success = {"train": 0, "val": 0, "test": 0}

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
                split_success[split_name] += 1
            except ValueError as exc:
                failures.append(
                    {
                        "image_path": str(image_path.resolve()),
                        "class_name": class_name,
                        "split": split_name,
                        "reason": str(exc),
                    }
                )

    if not rows:
        raise RuntimeError("Nenhuma feature válida foi gerada.")

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    failure_reasons = Counter(item["reason"] for item in failures)
    split_stats = {
        split: {
            "total_images": split_totals[split],
            "success": split_success[split],
            "failures": split_totals[split] - split_success[split],
        }
        for split in ("train", "val", "test")
    }

    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset_root": str(dataset_root.resolve()),
        "model_path": str(model_path.resolve()),
        "output_csv": str(output_csv.resolve()),
        "rows_generated": int(len(df)),
        "feature_count": int(len(df.columns) - 3),
        "splits": split_stats,
        "failures": {
            "total": int(len(failures)),
            "by_reason": {reason: int(count) for reason, count in failure_reasons.items()},
            "samples": failures[:20],
        },
    }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Geração de features geométricas concluída")
    print(f"- output_csv: {output_csv}")
    print(f"- output_report: {output_report}")
    print(f"- linhas geradas: {len(df)}")
    print(f"- falhas: {len(failures)}")
    if failures:
        print("- exemplo de falha:")
        sample = failures[0]
        print(f"  - {sample['image_path']} :: {sample['reason']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
