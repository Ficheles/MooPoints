import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.classification.inference_pipeline import KEYPOINT_MAP


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predição de keypoints de vaca a partir de uma imagem usando YOLO Pose.",
    )
    parser.add_argument(
        "--image-path",
        required=True,
        help="Caminho da imagem de entrada.",
    )
    parser.add_argument(
        "--model-path",
        default="models/yolo/best.pt",
        help="Caminho do modelo YOLO Pose (padrão: models/yolo/best.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confiança mínima para predição.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Tamanho da imagem para inferência.",
    )
    parser.add_argument(
        "--save-image",
        nargs="?",
        const="__AUTO__",
        default="",
        help="Caminho opcional para salvar imagem anotada com keypoints. Se informado sem valor, usa caminho automático.",
    )
    parser.add_argument(
        "--save-json",
        nargs="?",
        const="__AUTO__",
        default="",
        help="Caminho opcional para salvar keypoints em JSON. Se informado sem valor, usa caminho automático.",
    )
    return parser.parse_args()


def draw_keypoints(image_bgr, keypoints):
    output = image_bgr.copy()
    for i, point in enumerate(keypoints):
        x, y = int(point[0]), int(point[1])
        cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
        label = KEYPOINT_MAP.get(i, str(i))
        cv2.putText(output, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return output


def main():
    args = parse_args()

    image_path = Path(args.image_path)
    model_path = Path(args.model_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_path),
        task="pose",
        conf=args.conf,
        imgsz=args.imgsz,
        verbose=False,
    )

    if not results:
        raise RuntimeError("Nenhum resultado retornado pelo modelo.")

    keypoints_obj = getattr(results[0], "keypoints", None)
    if keypoints_obj is None or keypoints_obj.xy is None:
        raise RuntimeError("Nenhum keypoint detectado na imagem.")

    keypoints_batch = keypoints_obj.xy.cpu().numpy()
    if len(keypoints_batch) == 0:
        raise RuntimeError("Nenhum keypoint detectado na imagem.")

    keypoints = keypoints_batch[0]
    required_kpts = len(KEYPOINT_MAP)
    keypoints = keypoints[:required_kpts]

    payload = {
        "image_path": str(image_path.resolve()),
        "model_path": str(model_path.resolve()),
        "num_keypoints": int(len(keypoints)),
        "keypoints": [
            {
                "index": int(i),
                "name": KEYPOINT_MAP.get(i, str(i)),
                "x": float(point[0]),
                "y": float(point[1]),
            }
            for i, point in enumerate(keypoints)
        ],
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    auto_dir = Path("outputs") / "keypoints"
    auto_dir.mkdir(parents=True, exist_ok=True)

    save_json_value = args.save_json
    save_image_value = args.save_image

    if save_json_value == "__AUTO__":
        save_json_value = str(auto_dir / f"{image_path.stem}_keypoints.json")
    if save_image_value == "__AUTO__":
        save_image_value = str(auto_dir / f"{image_path.stem}_keypoints.jpg")

    if save_json_value:
        save_json_path = Path(save_json_value)
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        save_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON salvo em: {save_json_path}")

    if save_image_value:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Falha ao abrir imagem para anotação: {image_path}")
        annotated = draw_keypoints(image_bgr, keypoints)
        save_image_path = Path(save_image_value)
        save_image_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(save_image_path), annotated)
        if not ok:
            raise RuntimeError(f"Falha ao salvar imagem anotada em: {save_image_path}")
        print(f"Imagem anotada salva em: {save_image_path}")


if __name__ == "__main__":
    main()
