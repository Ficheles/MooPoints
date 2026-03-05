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
    0: 'Head', 1: 'Neck', 2: 'Withers', 3: 'Back', 4: 'Hook',
    5: 'Hip_ridge', 6: 'Tail_head', 7: 'Pin'
}
DISTANCE_PAIRS = [
    ('Head', 'Neck'), ('Neck', 'Withers'), ('Withers', 'Back'), ('Back', 'Hook'),
    ('Hook', 'Hip_ridge'), ('Hip_ridge', 'Tail_head'), ('Tail_head', 'Pin')
]
ANGLE_TRIPLETS = [
    ('Head', 'Neck', 'Withers'), ('Neck', 'Withers', 'Back'), ('Withers', 'Back', 'Hook'),
    ('Back', 'Hook', 'Hip_ridge'), ('Hook', 'Hip_ridge', 'Tail_head'), ('Hip_ridge', 'Tail_head', 'Pin')
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

        keypoints = {KEYPOINT_MAP[i]: coord for i, coord in enumerate(keypoints_data[0])}
        
        features = {}
        for p_name1, p_name2 in DISTANCE_PAIRS:
            features[f"dist_{p_name1}_{p_name2}"] = calculate_distance(keypoints[p_name1], keypoints[p_name2])
        
        for p_name1, p_name2, p_name3 in ANGLE_TRIPLETS:
            features[f"angle_{p_name1}_{p_name2}_{p_name3}"] = calculate_angle(keypoints[p_name1], keypoints[p_name2], keypoints[p_name3])

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
