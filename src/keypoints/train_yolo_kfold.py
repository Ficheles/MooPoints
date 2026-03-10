import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from statistics import mean

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treinamento YOLO Pose com K-Fold usando dataset já preparado.",
    )
    parser.add_argument(
        "--dataset-root",
        default=os.getenv("DATASET_ROOT", "data/datasets/keypoints"),
        help="Diretório raiz com fold_*/data_fold_*.yaml",
    )
    parser.add_argument(
        "--models-dir",
        default=os.getenv("MODEL_DIR", "models/yolo"),
        help="Diretório com pesos base de transfer learning",
    )
    parser.add_argument(
        "--base-model",
        default=os.getenv("TRAIN_MODEL_NAME", "yolo11x-pose.pt"),
        help="Nome do arquivo de pesos base dentro de --models-dir",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.getenv("EPOCHS", "100")),
        help="Número de épocas por fold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(os.getenv("IMGSZ", "640")),
        help="Tamanho da imagem para treino",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=int(os.getenv("BATCH", "8")),
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("DEVICE", "cpu"),
        help="Dispositivo de treino (ex: cpu, 0)",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("TRAIN_PROJECT", "runs/kfold_pose"),
        help="Diretório base para saída dos treinos",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("WORKERS", "2")),
        help="Quantidade de workers do dataloader",
    )
    parser.add_argument(
        "--run-prefix",
        default=os.getenv("RUN_PREFIX", "train"),
        help="Prefixo do nome da execução por fold",
    )
    parser.add_argument(
        "--continue-from-best",
        action="store_true",
        default=os.getenv("CONTINUE_FROM_BEST", "false").lower() == "true",
        help="Continua o treino de cada fold a partir do best.pt salvo anteriormente",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=int(os.getenv("PATIENCE", "50")),
        help="Early stopping patience",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=int(os.getenv("SAVE_PERIOD", "10")),
        help="Intervalo (épocas) para salvar checkpoints intermediários",
    )
    parser.add_argument(
        "--conf-min",
        type=float,
        default=float(os.getenv("CONF_MIN", "0.30")),
        help="Confiança mínima usada no bloco de métricas com rejeição (template)",
    )
    return parser.parse_args()


def find_fold_yaml_files(dataset_root: Path):
    fold_yaml_files = []
    for fold_dir in sorted(dataset_root.glob("fold_*")):
        if not fold_dir.is_dir():
            continue

        yaml_candidates = sorted(fold_dir.glob("data_fold_*.yaml")) + sorted(fold_dir.glob("data_fold_*.yml"))
        if yaml_candidates:
            fold_yaml_files.append((fold_dir.name, yaml_candidates[0]))

    return fold_yaml_files


def read_metric(train_result, metric_key):
    results_dict = getattr(train_result, "results_dict", None)
    if isinstance(results_dict, dict):
        value = results_dict.get(metric_key)
        if value is not None:
            return value
    return None


def read_first_metric(train_result, keys):
    for key in keys:
        value = read_metric(train_result, key)
        if value is not None:
            return value
    return None


def existing_best_checkpoint(project_dir: Path, run_name: str):
    best_path = project_dir / run_name / "weights" / "best.pt"
    return best_path if best_path.exists() else None


def read_best_epoch_stats(results_csv_path: Path):
    if not results_csv_path.exists():
        return {}

    with results_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {}

    pose_key = "metrics/mAP50-95(P)"
    box_key = "metrics/mAP50-95(B)"
    selected_key = pose_key if pose_key in rows[0] else box_key

    best_row = None
    best_value = float("-inf")
    for row in rows:
        raw = row.get(selected_key)
        if raw in (None, ""):
            continue
        value = float(raw)
        if value > best_value:
            best_value = value
            best_row = row

    if best_row is None:
        return {}

    return {
        "best_epoch": int(float(best_row.get("epoch", 0))),
        "metric_used": selected_key,
        "metric_value": best_value,
    }


def mean_or_none(values):
    valid = [v for v in values if isinstance(v, (int, float))]
    return mean(valid) if valid else None


def build_report(summary, conf_min):
    box_map50_values = [item.get("box_map50") for item in summary]
    box_map5095_values = [item.get("box_map50_95") for item in summary]
    pose_map50_values = [item.get("pose_map50") for item in summary]
    pose_map5095_values = [item.get("pose_map50_95") for item in summary]

    pose_best_fold = None
    best_pose_value = float("-inf")
    for item in summary:
        value = item.get("pose_map50_95")
        if isinstance(value, (int, float)) and value > best_pose_value:
            best_pose_value = value
            pose_best_fold = item

    report = {
        "k_folds": len(summary),
        "aggregate_metrics": {
            "Box_mAP50_media_folds": mean_or_none(box_map50_values),
            "Box_mAP50_95_media_folds": mean_or_none(box_map5095_values),
            "Pose_mAP50_media_folds": mean_or_none(pose_map50_values),
            "Pose_mAP50_95_media_folds": mean_or_none(pose_map5095_values),
            "Pose_mAP50_95_melhor_fold": {
                "valor": None if pose_best_fold is None else pose_best_fold.get("pose_map50_95"),
                "fold": None if pose_best_fold is None else pose_best_fold.get("fold"),
            },
        },
        "metricas_teste_final": {
            "accuracy": None,
            "f1_macro": None,
            "top1_accuracy": None,
            "top3_accuracy": None,
            "top5_accuracy": None,
        },
        "metricas_com_rejeicao": {
            "confianca_min": conf_min,
            "cobertura": None,
            "accuracy_aceitas": None,
            "f1_macro_aceitas": None,
        },
        "folds": summary,
    }
    return report


def print_report_console(report):
    metrics = report["aggregate_metrics"]
    best_fold = metrics["Pose_mAP50_95_melhor_fold"]

    print("\n" + "=" * 60)
    print(f"k_folds: {report['k_folds']}")
    print(f"Box_mAP50 (média dos folds): {metrics['Box_mAP50_media_folds']}")
    print(f"Box_mAP50-95 (média dos folds): {metrics['Box_mAP50_95_media_folds']}")
    print(f"Pose_mAP50 (média dos folds): {metrics['Pose_mAP50_media_folds']}")
    print(f"Pose_mAP50-95 (média dos folds): {metrics['Pose_mAP50_95_media_folds']}")
    print(
        "Pose_mAP50-95 (melhor fold): "
        f"{best_fold['valor']} ({best_fold['fold']})"
    )
    print("=" * 60)


def train_kfold(args):
    dataset_root = Path(args.dataset_root)
    models_dir = Path(args.models_dir)
    base_model_path = models_dir / args.base_model

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root não encontrado: {dataset_root}")

    if not base_model_path.exists():
        raise FileNotFoundError(f"Modelo base não encontrado: {base_model_path}")

    fold_yaml_files = find_fold_yaml_files(dataset_root)
    if not fold_yaml_files:
        raise FileNotFoundError(
            f"Nenhum data_fold_*.yaml encontrado em {dataset_root}/fold_*"
        )

    project_dir = Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Treino YOLO Pose com K-Fold")
    print(f"Dataset root: {dataset_root}")
    print(f"Modelo base: {base_model_path}")
    print(f"Total de folds: {len(fold_yaml_files)}")
    print("=" * 50)

    summary = []

    for fold_name, data_yaml in fold_yaml_files:
        run_name = f"{args.run_prefix}_{fold_name}"
        print(f"\n--- Iniciando {fold_name} ---")
        print(f"YAML: {data_yaml}")
        model_start_path = base_model_path
        best_checkpoint = existing_best_checkpoint(project_dir, run_name)
        if args.continue_from_best and best_checkpoint is not None:
            model_start_path = best_checkpoint
            print(f"Continuando do melhor checkpoint salvo: {model_start_path}")
        else:
            print(f"Iniciando com modelo base: {model_start_path}")

        model = YOLO(str(model_start_path))
        result = model.train(
            data=str(data_yaml),
            task="pose",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=str(project_dir),
            name=run_name,
            save=True,
            save_period=args.save_period,
            patience=args.patience,
            exist_ok=True,
        )

        box_map50 = read_first_metric(result, ["metrics/mAP50(B)"])
        box_map5095 = read_first_metric(result, ["metrics/mAP50-95(B)"])
        pose_map50 = read_first_metric(result, ["metrics/mAP50(P)"])
        pose_map5095 = read_first_metric(result, ["metrics/mAP50-95(P)"])

        save_dir = Path(getattr(result, "save_dir", project_dir / run_name))
        best_weights = save_dir / "weights" / "best.pt"
        last_weights = save_dir / "weights" / "last.pt"
        best_epoch_stats = read_best_epoch_stats(save_dir / "results.csv")

        summary.append(
            {
                "fold": fold_name,
                "run": run_name,
                "box_map50": box_map50,
                "box_map50_95": box_map5095,
                "pose_map50": pose_map50,
                "pose_map50_95": pose_map5095,
                "best_weights": str(best_weights) if best_weights.exists() else None,
                "last_weights": str(last_weights) if last_weights.exists() else None,
                "best_epoch_stats": best_epoch_stats,
            }
        )

    report = build_report(summary, args.conf_min)
    print_report_console(report)

    report_path = project_dir / "kfold_metrics_report.json"
    report_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "settings": {
            "dataset_root": str(dataset_root),
            "base_model": str(base_model_path),
            "continue_from_best": args.continue_from_best,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "project": str(project_dir),
            "run_prefix": args.run_prefix,
            "save_period": args.save_period,
            "patience": args.patience,
        },
        "report": report,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)

    print(f"\nRelatório salvo em: {report_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    train_kfold(cli_args)
