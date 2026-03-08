import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferência de classificação de vaca (top-1 e top-k) para uma imagem.",
    )
    parser.add_argument("image_path", help="Caminho da imagem de entrada.")
    parser.add_argument(
        "--model-path",
        default="runs/kfold_classification/cls_fold_0/weights/best.pt",
        help="Checkpoint do modelo treinado (best.pt).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Quantidade de classes para exibir no ranking.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image_path)
    model_path = Path(args.model_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    model = YOLO(str(model_path))
    results = model.predict(source=str(image_path), task="classify", verbose=False)

    if not results:
        print("Nenhum resultado retornado pelo classificador.")
        return

    probs = results[0].probs
    top1_index = int(probs.top1)
    top1_conf = float(probs.top1conf)
    names = model.names

    print("=" * 60)
    print(f"Imagem: {image_path}")
    print(f"Classe prevista (top-1): {names[top1_index]}")
    print(f"Confiança top-1: {top1_conf:.6f}")
    print("\nRanking top-k:")

    topk = min(5, max(1, args.topk))
    top_indices = probs.top5[:topk]
    top_conf = probs.top5conf[:topk]

    for rank, (class_idx, conf) in enumerate(zip(top_indices, top_conf), start=1):
        print(f"{rank:>2}) {names[int(class_idx)]}: {float(conf):.6f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
