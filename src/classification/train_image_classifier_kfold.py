import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from statistics import mean

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina classificador de imagem com K-Fold em data/datasets/classifications/fold_*.",
    )
    parser.add_argument(
        "--dataset-root",
        default=os.getenv("DATASET_CLASSIFICATION_ROOT", "data/datasets/classifications"),
        help="Diretório raiz contendo fold_*/train e fold_*/val.",
    )
    parser.add_argument(
        "--models-dir",
        default=os.getenv("MODEL_DIR", "models/yolo"),
        help="Diretório para pesos base.",
    )
    parser.add_argument(
        "--base-model",
        default=os.getenv("CLS_BASE_MODEL", "yolo11n-cls.pt"),
        help="Modelo base de classificação (ex: yolo11n-cls.pt).",
    )
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "50")))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("IMGSZ", "640")))
    parser.add_argument("--batch", type=int, default=int(os.getenv("BATCH", "16")))
    parser.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "4")))
    parser.add_argument("--device", default=os.getenv("DEVICE", "cpu"))
    parser.add_argument(
        "--project",
        default=os.getenv("TRAIN_PROJECT", "runs/kfold_classification"),
    )
    parser.add_argument(
        "--run-prefix",
        default=os.getenv("RUN_PREFIX", "cls"),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=int(os.getenv("PATIENCE", "20")),
    )
    return parser.parse_args()


def mean_or_none(values):
    valid = [v for v in values if isinstance(v, (int, float))]
    return mean(valid) if valid else None


def safe_metric(metrics_obj, keys):
    results_dict = getattr(metrics_obj, "results_dict", None)
    if not isinstance(results_dict, dict):
        return None
    for key in keys:
        value = results_dict.get(key)
        if value is not None:
            return value
    return None


def find_folds(dataset_root: Path):
    folds = []
    for fold_dir in sorted(dataset_root.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        if (fold_dir / "train").exists() and (fold_dir / "val").exists():
            folds.append(fold_dir)
    return folds


def train_kfold(args):
    dataset_root = Path(args.dataset_root)
    models_dir = Path(args.models_dir)
    project_dir = Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root não encontrado: {dataset_root}")

    base_model_path = models_dir / args.base_model
    model_source = str(base_model_path) if base_model_path.exists() else args.base_model

    folds = find_folds(dataset_root)
    if not folds:
        raise FileNotFoundError(
            f"Nenhum fold válido encontrado em {dataset_root} (esperado fold_*/train e fold_*/val)."
        )

    print("=" * 70)
    print("Treinamento K-Fold - Classificação de 30 classes")
    print(f"dataset_root: {dataset_root.resolve()}")
    print(f"base_model: {model_source}")
    print(f"folds: {[f.name for f in folds]}")
    print("=" * 70)

    summary = []

    for fold_dir in folds:
        run_name = f"{args.run_prefix}_{fold_dir.name}"
        print(f"\n--- Treinando {fold_dir.name} ---")

        model = YOLO(model_source)
        train_result = model.train(
            task="classify",
            data=str(fold_dir),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=str(project_dir),
            name=run_name,
            patience=args.patience,
            exist_ok=True,
        )

        save_dir = Path(getattr(train_result, "save_dir", project_dir / run_name))
        best_weights = save_dir / "weights" / "best.pt"

        top1_train = safe_metric(train_result, ["metrics/accuracy_top1", "metrics/top1"])
        top5_train = safe_metric(train_result, ["metrics/accuracy_top5", "metrics/top5"])

        top1_val = None
        top5_val = None
        if best_weights.exists():
            val_model = YOLO(str(best_weights))
            val_result = val_model.val(
                task="classify",
                data=str(fold_dir),
                split="val",
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                device=args.device,
                verbose=False,
            )
            top1_val = safe_metric(val_result, ["metrics/accuracy_top1", "metrics/top1"])
            top5_val = safe_metric(val_result, ["metrics/accuracy_top5", "metrics/top5"])

        summary.append(
            {
                "fold": fold_dir.name,
                "run": run_name,
                "top1_train": top1_train,
                "top5_train": top5_train,
                "top1_val": top1_val,
                "top5_val": top5_val,
                "best_weights": str(best_weights) if best_weights.exists() else None,
            }
        )

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset_root": str(dataset_root.resolve()),
        "base_model": model_source,
        "settings": {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "project": str(project_dir.resolve()),
            "run_prefix": args.run_prefix,
            "patience": args.patience,
        },
        "aggregate": {
            "top1_val_mean": mean_or_none([item.get("top1_val") for item in summary]),
            "top5_val_mean": mean_or_none([item.get("top5_val") for item in summary]),
            "top1_train_mean": mean_or_none([item.get("top1_train") for item in summary]),
            "top5_train_mean": mean_or_none([item.get("top5_train") for item in summary]),
        },
        "folds": summary,
    }

    report_path = project_dir / "kfold_classification_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nResumo final:")
    print(f"- top1_val_mean: {report['aggregate']['top1_val_mean']}")
    print(f"- top5_val_mean: {report['aggregate']['top5_val_mean']}")
    print(f"Relatório salvo em: {report_path}")


if __name__ == "__main__":
    args = parse_args()
    train_kfold(args)
