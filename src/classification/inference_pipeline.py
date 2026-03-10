"""
Módulo de Inferência End-to-End para Identificação de Bovinos.

Este script carrega o modelo YOLO-Pose treinado e o classificador XGBoost
para realizar a predição da identidade de uma vaca (cow_id) a partir de uma
única imagem.

O pipeline executa os seguintes passos:
1. Carrega uma imagem de entrada.
2. Usa o modelo YOLO para detectar os 8 pontos-chave.
3. Calcula as features geométricas (distâncias e ângulos) a partir dos pontos.
4. Usa o classificador XGBoost para prever a 'cow_id'.

Autor: GitHub Copilot
Data: 2026-03-05
"""
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from ultralytics import YOLO

# --- Constantes e Mapeamentos ---
# (Reutilizados de extract_geometric_features.py para consistência)
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
DISTANCE_PAIRS = [
    ("withers", "back"),
    ("back", "hook up"),
    ("hook up", "hook down"),
    ("hook down", "hip"),
    ("hip", "tail head"),
    ("tail head", "pin up"),
    ("pin up", "pin down"),
]
ANGLE_TRIPLETS = [
    ("withers", "back", "hook up"),
    ("back", "hook up", "hook down"),
    ("hook up", "hook down", "hip"),
    ("hook down", "hip", "tail head"),
    ("hip", "tail head", "pin up"),
    ("tail head", "pin up", "pin down"),
]

# --- Funções de Cálculo ---
def calculate_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    angle = np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))
    angle = np.abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle


def _slug(name: str) -> str:
    return name.replace(" ", "_")


def build_feature_dict(keypoints: np.ndarray) -> dict[str, float]:
    required_kpts = len(KEYPOINT_MAP)
    if keypoints.shape[0] < required_kpts:
        raise ValueError(f"Esperava {required_kpts} pontos-chave, mas recebeu {keypoints.shape[0]}")

    keypoints = keypoints[:required_kpts]
    keypoint_map = {KEYPOINT_MAP[i]: keypoints[i] for i in range(required_kpts)}
    ordered_names = [KEYPOINT_MAP[i] for i in range(required_kpts)]

    features: dict[str, float] = {}

    for p_name1, p_name2 in DISTANCE_PAIRS:
        features[f"dist_{p_name1}_{p_name2}"] = float(calculate_distance(keypoint_map[p_name1], keypoint_map[p_name2]))

    for p_name1, p_name2, p_name3 in ANGLE_TRIPLETS:
        features[f"angle_{p_name1}_{p_name2}_{p_name3}"] = float(calculate_angle(keypoint_map[p_name1], keypoint_map[p_name2], keypoint_map[p_name3]))

    for i in range(required_kpts):
        for j in range(i + 1, required_kpts):
            n1 = ordered_names[i]
            n2 = ordered_names[j]
            col = f"dist_pair_{_slug(n1)}_{_slug(n2)}"
            features[col] = float(calculate_distance(keypoint_map[n1], keypoint_map[n2]))

    xs = keypoints[:, 0]
    ys = keypoints[:, 1]
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    bbox_w = float(x_max - x_min)
    bbox_h = float(y_max - y_min)
    bbox_diag = float(np.hypot(bbox_w, bbox_h))
    bbox_area = float(max(0.0, bbox_w * bbox_h))
    bbox_aspect = float(bbox_w / (bbox_h + 1e-6))

    centroid = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=float)
    dists_centroid = np.linalg.norm(keypoints - centroid, axis=1)

    features["kp_bbox_w"] = bbox_w
    features["kp_bbox_h"] = bbox_h
    features["kp_bbox_diag"] = bbox_diag
    features["kp_bbox_area"] = bbox_area
    features["kp_bbox_aspect"] = bbox_aspect
    features["kp_centroid_x"] = float(centroid[0])
    features["kp_centroid_y"] = float(centroid[1])
    features["kp_dist_centroid_mean"] = float(np.mean(dists_centroid))
    features["kp_dist_centroid_std"] = float(np.std(dists_centroid))
    features["kp_dist_centroid_min"] = float(np.min(dists_centroid))
    features["kp_dist_centroid_max"] = float(np.max(dists_centroid))

    norm_ref = bbox_diag + 1e-6
    for p_name1, p_name2 in DISTANCE_PAIRS:
        base_col = f"dist_{p_name1}_{p_name2}"
        features[f"{base_col}_norm_diag"] = float(features[base_col] / norm_ref)

    for i in range(required_kpts - 1):
        n1 = ordered_names[i]
        n2 = ordered_names[i + 1]
        p1 = keypoint_map[n1]
        p2 = keypoint_map[n2]
        theta = float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))
        features[f"orient_{_slug(n1)}_{_slug(n2)}"] = theta

    return features

# --- Classe do Pipeline ---
class InferencePipeline:
    def __init__(self, yolo_model_path, xgb_model_path, encoder_path):
        print("Carregando modelos...")
        self.yolo_model = YOLO(yolo_model_path)
        self.xgb_model = joblib.load(xgb_model_path)
        self.label_encoder = joblib.load(encoder_path)
        print("Modelos carregados com sucesso.")

    def extract_features(self, keypoints_data):
        """Extrai features geométricas a partir das coordenadas dos keypoints."""
        if keypoints_data.shape[1] < 8:
             raise ValueError(f"Esperava 8 pontos-chave, mas recebeu {keypoints_data.shape[1]}")
        features = build_feature_dict(keypoints_data[0])
        return pd.DataFrame([features])

    def predict(self, image_path):
        """Executa o pipeline completo para uma imagem."""
        print(f"\nProcessando imagem: {image_path}")

        # 1. Predição com YOLO-Pose
        results = self.yolo_model(image_path, verbose=False)
        if not results or len(results[0].keypoints.xy.cpu().numpy()) == 0:
            print("AVISO: Nenhuma vaca ou ponto-chave detectado na imagem.")
            return None

        keypoints_data = results[0].keypoints.xy.cpu().numpy()

        # 2. Extração de Features Geométricas
        try:
            features_df = self.extract_features(keypoints_data)
        except ValueError as e:
            print(f"ERRO: {e}")
            return None
            
        # 3. Predição com XGBoost
        prediction_encoded = self.xgb_model.predict(features_df.values)
        
        # 4. Decodificação do Resultado
        cow_id = self.label_encoder.inverse_transform(prediction_encoded)

        return cow_id[0]

def main():
    parser = argparse.ArgumentParser(description="Pipeline de inferência para identificação de vacas.")
    parser.add_argument("image_path", help="Caminho para a imagem de entrada.")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    yolo_model = project_root / 'models' / 'yolo11x-pose.pt' 
    xgb_model = project_root / 'models' / 'xgboost_cow_id.pkl'
    encoder = project_root / 'models' / 'label_encoder.pkl'

    if not all([yolo_model.exists(), xgb_model.exists(), encoder.exists()]):
        print("ERRO: Um ou mais arquivos de modelo não foram encontrados. Certifique-se de que os modelos estão em 'models/'.")
        return

    pipeline = InferencePipeline(yolo_model, xgb_model, encoder)
    predicted_id = pipeline.predict(args.image_path)

    if predicted_id:
        print("-" * 50)
        print(f">> Identidade da Vaca (cow_id): {predicted_id} <<")
        print("-" * 50)

if __name__ == '__main__':
    main()
