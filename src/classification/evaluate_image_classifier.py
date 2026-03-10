import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avalia classificador por imagem no conjunto data/datasets/classifications/test/images.",
    )
    parser.add_argument(
        "--test-root",
        default="data/datasets/classifications/test/images",
        help="Diretório de teste com subpastas por classe.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Checkpoint do classificador treinado (best.pt).",
    )
    parser.add_argument(
        "--output-json",
        default="runs/kfold_classification/test_metrics.json",
        help="Arquivo JSON de saída com métricas.",
    )
    return parser.parse_args()


def collect_test_samples(test_root: Path):
    samples = []
    class_dirs = sorted([d for d in test_root.iterdir() if d.is_dir()])
    for class_dir in class_dirs:
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, class_dir.name))
    return samples


def safe_topk_hit(prob_obj, true_index: int, k: int):
    if k <= 1:
        return int(int(prob_obj.top1) == true_index)
    topk_indices = [int(idx) for idx in prob_obj.top5][:k]
    return int(true_index in topk_indices)


def main():
    args = parse_args()
    test_root = Path(args.test_root)
    model_path = Path(args.model_path)
    output_path = Path(args.output_json)

    if not test_root.exists():
        raise FileNotFoundError(f"Conjunto de teste não encontrado: {test_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    samples = collect_test_samples(test_root)
    if not samples:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em: {test_root}")

    model = YOLO(str(model_path))
    name_to_idx = {name: int(idx) for idx, name in model.names.items()}

    y_true = []
    y_pred = []
    top3_hits = []
    top5_hits = []

    missing_classes = set()

    for image_path, class_name in samples:
        if class_name not in name_to_idx:
            missing_classes.add(class_name)
            continue

        true_idx = name_to_idx[class_name]
        results = model.predict(source=str(image_path), task="classify", verbose=False)
        if not results:
            continue

        probs = results[0].probs
        pred_idx = int(probs.top1)

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        top3_hits.append(safe_topk_hit(probs, true_idx=true_idx, k=3))
        top5_hits.append(safe_topk_hit(probs, true_idx=true_idx, k=5))

    if missing_classes:
        print("Aviso: classes de teste ausentes no modelo e ignoradas:")
        for class_name in sorted(missing_classes):
            print(f"- {class_name}")

    if not y_true:
        raise RuntimeError("Nenhuma amostra válida para avaliação.")

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    top3 = sum(top3_hits) / len(top3_hits)
    top5 = sum(top5_hits) / len(top5_hits)
    cm = confusion_matrix(y_true, y_pred).tolist()

    report = {
        "samples": len(y_true),
        "accuracy_top1": accuracy,
        "f1_macro": f1_macro,
        "accuracy_top3": top3,
        "accuracy_top5": top5,
        "model_path": str(model_path.resolve()),
        "test_root": str(test_root.resolve()),
        "confusion_matrix": cm,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print("Avaliação no conjunto de teste")
    print(f"- amostras: {report['samples']}")
    print(f"- accuracy_top1: {report['accuracy_top1']:.6f}")
    print(f"- f1_macro: {report['f1_macro']:.6f}")
    print(f"- accuracy_top3: {report['accuracy_top3']:.6f}")
    print(f"- accuracy_top5: {report['accuracy_top5']:.6f}")
    print(f"- relatório: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
