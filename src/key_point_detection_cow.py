import cv2
import os
from ultralytics import YOLO

IMAGE_PATH = "/app/dataset_samples/person.png"
SAVE_PATH = "/app/dataset_samples/person-pose_with_skeleton.png"
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
DETECTION_MODEL_NAME = os.getenv("DETECTION_MODEL_NAME", "yolo11x.pt")
POSE_MODEL_NAME = os.getenv("POSE_MODEL_NAME", "yolo11x-pose.pt")
DETECTION_MODEL = os.path.join(MODEL_DIR, DETECTION_MODEL_NAME)
POSE_MODEL = os.path.join(MODEL_DIR, POSE_MODEL_NAME)


def draw_pose(image, keypoints_xy, keypoints_conf, thickness=2):
    if keypoints_xy is None or len(keypoints_xy) == 0 or keypoints_conf is None:
        return image

    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    for kpts, confs in zip(keypoints_xy, keypoints_conf):
        kpts = kpts.cpu().numpy()
        confs = confs.cpu().numpy()

        for index, (x, y) in enumerate(kpts):
            if confs[index] > 0.5:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

        for start, end in skeleton:
            if confs[start] > 0.5 and confs[end] > 0.5:
                start_pt = (int(kpts[start][0]), int(kpts[start][1]))
                end_pt = (int(kpts[end][0]), int(kpts[end][1]))
                cv2.line(image, start_pt, end_pt, (255, 0, 0), thickness)

    return image


def main():
    det_model = YOLO(DETECTION_MODEL)
    pose_model = YOLO(POSE_MODEL)

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    det_results = det_model(
        image,
        conf=0.25,
        iou=0.45,
        classes=[0],
        verbose=False,
    )

    vis_image = image.copy()
    for result in det_results:
        if result.boxes is not None and result.boxes.xyxy is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    vis_image,
                    "Person",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    3,
                )

    pose_results = pose_model(
        image,
        conf=0.25,
        iou=0.45,
        classes=[0],
        verbose=False,
    )

    keypoints_xy = []
    keypoints_conf = []
    for result in pose_results:
        if result.keypoints is not None:
            keypoints_xy = result.keypoints.xy
            keypoints_conf = result.keypoints.conf

    vis_image = draw_pose(vis_image, keypoints_xy, keypoints_conf)
    cv2.imwrite(SAVE_PATH, vis_image)
    print(f"Output image saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
