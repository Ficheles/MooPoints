import argparse
import json
from pathlib import Path


KEYPOINT_ORDER = [
    "withers",
    "back",
    "hook up",
    "hook down",
    "hip",
    "tail head",
    "pin up",
    "pin down",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Converte labels JSON (Label Studio) para formato YOLO Pose.",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/datasets/keypoints",
        help="Diretório raiz contendo fold_*/labels/*/*.json",
    )
    return parser.parse_args()


def clamp_01(value):
    return max(0.0, min(1.0, float(value)))


def visibility_from_choices(choices):
    if not choices:
        return 2
    text = " ".join(str(item).lower() for item in choices)
    if "invis" in text or "ocult" in text:
        return 1
    return 2


def convert_result_array_to_yolo_pose(annotation_data):
    results = annotation_data.get("result", [])
    if not isinstance(results, list) or not results:
        return None

    visibility_by_id = {}
    for res in results:
        if res.get("type") != "choices":
            continue
        value = res.get("value", {})
        visibility_by_id[res.get("id")] = visibility_from_choices(value.get("choices", []))

    bbox = None
    keypoints = {}

    for res in results:
        res_type = res.get("type")
        value = res.get("value", {})

        if res_type == "rectanglelabels" and bbox is None:
            x = clamp_01(value.get("x", 0.0) / 100.0)
            y = clamp_01(value.get("y", 0.0) / 100.0)
            w = clamp_01(value.get("width", 0.0) / 100.0)
            h = clamp_01(value.get("height", 0.0) / 100.0)
            xc = clamp_01(x + (w / 2.0))
            yc = clamp_01(y + (h / 2.0))
            bbox = (xc, yc, w, h)

        if res_type == "keypointlabels":
            labels = value.get("keypointlabels", [])
            if not labels:
                continue
            kp_name = labels[0]
            if kp_name not in KEYPOINT_ORDER:
                continue
            x = clamp_01(value.get("x", 0.0) / 100.0)
            y = clamp_01(value.get("y", 0.0) / 100.0)
            v = visibility_by_id.get(res.get("id"), 2)
            keypoints[kp_name] = (x, y, v)

    if bbox is None and keypoints:
        xs = [kp[0] for kp in keypoints.values()]
        ys = [kp[1] for kp in keypoints.values()]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        w = clamp_01(xmax - xmin)
        h = clamp_01(ymax - ymin)
        xc = clamp_01((xmin + xmax) / 2.0)
        yc = clamp_01((ymin + ymax) / 2.0)
        bbox = (xc, yc, w, h)

    if bbox is None:
        return None

    xc, yc, w, h = bbox
    line_values = [0, xc, yc, w, h]

    for kp_name in KEYPOINT_ORDER:
        kp = keypoints.get(kp_name)
        if kp is None:
            line_values.extend([0.0, 0.0, 0])
        else:
            line_values.extend([kp[0], kp[1], kp[2]])

    yolo_tokens = [
        str(line_values[0]),
        *[f"{float(v):.6f}" if isinstance(v, float) else str(v) for v in line_values[1:]],
    ]
    return " ".join(yolo_tokens)


def convert_label_file_if_needed(json_label_path: Path):
    raw = json_label_path.read_text(encoding="utf-8").strip()
    if not raw:
        return False, "empty"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False, "invalid_json"

    yolo_line = convert_result_array_to_yolo_pose(data)
    if yolo_line is None:
        return False, "missing_result_or_bbox"

    txt_label_path = json_label_path.with_suffix(".txt")
    txt_label_path.write_text(yolo_line + "\n", encoding="utf-8")
    return True, "converted"


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root não encontrado: {dataset_root}")

    stats = {
        "converted": 0,
        "empty": 0,
        "invalid_json": 0,
        "missing_result_or_bbox": 0,
    }

    label_files = list(dataset_root.glob("fold_*/labels/*/*.json"))
    if not label_files:
        print(f"Nenhum label encontrado em: {dataset_root}/fold_*/labels/*/*.json")
        return

    for label_file in label_files:
        converted, reason = convert_label_file_if_needed(label_file)
        if converted:
            stats["converted"] += 1
        else:
            stats[reason] = stats.get(reason, 0) + 1

    print("\nConversão de labels para YOLO Pose:")
    print(f"- dataset_root: {dataset_root}")
    print(f"- total_labels: {len(label_files)}")
    for key, value in stats.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
