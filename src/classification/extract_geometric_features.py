"""
Módulo para Extração de Features Geométricas de Anotações YOLO-Pose.

Este script lê os arquivos de anotação de keypoints no formato YOLO, calcula
distâncias euclidianas e ângulos entre os pontos-chave definidos, e salva
o resultado em um arquivo CSV tabular.

Autor: GitHub Copilot
Data: 2026-03-05
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# --- Constantes e Mapeamentos ---

# Mapeia o índice do keypoint para uma parte anatômica da vaca.
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

# Pares de pontos para cálculo de distância.
DISTANCE_PAIRS = [
    ("withers", "back"),
    ("back", "hook up"),
    ("hook up", "hook down"),
    ("hook down", "hip"),
    ("hip", "tail head"),
    ("tail head", "pin up"),
    ("pin up", "pin down"),
]

# Tripletos de pontos para cálculo de ângulo. O ponto do meio é o vértice.
ANGLE_TRIPLETS = [
    ("withers", "back", "hook up"),
    ("back", "hook up", "hook down"),
    ("hook up", "hook down", "hip"),
    ("hook down", "hip", "tail head"),
    ("hip", "tail head", "pin up"),
    ("tail head", "pin up", "pin down"),
]


# --- Funções de Cálculo ---

def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calcula a distância Euclidiana entre dois pontos."""
    return np.linalg.norm(p1 - p2)

def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calcula o ângulo no vértice p2 formado pelos pontos p1 e p3.

    A fórmula utiliza arctan2 para robustez e o resultado é normalizado
    para o intervalo [0, 180].
    """
    v1 = p1 - p2
    v2 = p3 - p2
    angle = np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))
    # Normaliza o ângulo para garantir que seja sempre o menor ângulo (< 180)
    angle = np.abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle

def extract_cow_id_from_filename(filename: str) -> str:
    """
    Extrai o ID da vaca a partir do nome do arquivo.
    Assume o padrão 'cow_id_YYYY_MM_DD...'.
    """
    try:
        # Pega o nome base do arquivo (sem extensão) e divide por '_'
        base_name = Path(filename).stem
        cow_id = base_name.split('_')[0]
        return cow_id
    except IndexError:
        # Retorna None se o padrão não for encontrado
        return None

# --- Processamento Principal ---

def process_yolo_annotations(labels_dir: str) -> pd.DataFrame:
    """
    Processa todos os arquivos de anotação em um diretório, extrai features
    geométricas e retorna um DataFrame consolidado.
    """
    # Encontra todos os arquivos de anotação .txt recursivamente
    search_pattern = os.path.join(labels_dir, 'fold_*', 'labels', '*', '*.txt')
    annotation_files = glob.glob(search_pattern)

    if not annotation_files:
        print(f"Aviso: Nenhum arquivo de anotação encontrado em '{search_pattern}'.")
        return pd.DataFrame()

    all_features = []
    
    print(f"Encontrados {len(annotation_files)} arquivos de anotação. Processando...")

    for filepath in annotation_files:
        # Extrai cow_id do nome do arquivo
        cow_id = extract_cow_id_from_filename(filepath)
        if cow_id is None:
            print(f"Aviso: Não foi possível extrair cow_id do arquivo {filepath}. Pulando.")
            continue

        with open(filepath, 'r') as f:
            line = f.readline()

        parts = line.strip().split(' ')
        # Os keypoints começam após as 5 primeiras colunas (class, x, y, w, h)
        keypoints_raw = np.array([float(p) for p in parts[5:]]).reshape(-1, 2)

        # Mapeia os keypoints lidos para um dicionário
        keypoints = {}
        for i, name in KEYPOINT_MAP.items():
            if i < len(keypoints_raw):
                # Pontos não detectados pelo YOLO são marcados como (0,0).
                # Substituímos por NaN para tratamento adequado.
                if np.all(keypoints_raw[i] == 0):
                    keypoints[name] = np.array([np.nan, np.nan])
                else:
                    keypoints[name] = keypoints_raw[i]
            else:
                keypoints[name] = np.array([np.nan, np.nan])

        # Calcula as features para a amostra atual
        features = {'filename': Path(filepath).name, 'cow_id': cow_id}

        # Calcula distâncias
        for p_name1, p_name2 in DISTANCE_PAIRS:
            col_name = f"dist_{p_name1}_{p_name2}"
            p1, p2 = keypoints[p_name1], keypoints[p_name2]
            if np.isnan(p1).any() or np.isnan(p2).any():
                features[col_name] = np.nan
            else:
                features[col_name] = calculate_distance(p1, p2)

        # Calcula ângulos
        for p_name1, p_name2, p_name3 in ANGLE_TRIPLETS:
            col_name = f"angle_{p_name1}_{p_name2}_{p_name3}"
            p1, p2, p3 = keypoints[p_name1], keypoints[p_name2], keypoints[p_name3]
            if np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any():
                features[col_name] = np.nan
            else:
                features[col_name] = calculate_angle(p1, p2, p3)

        all_features.append(features)

    return pd.DataFrame(all_features)


def main():
    """
    Função principal para orquestrar a extração de features e salvar o resultado.
    """
    print("Iniciando o script de extração de features geométricas...")
    
    # O script está em 'src/', então o diretório 'dataset' está um nível acima.
    project_root = Path(__file__).parent.parent
    labels_directory = project_root / 'dataset'
    output_directory = project_root / 'dataset' / 'data'
    
    # Cria o diretório de saída se ele não existir
    output_directory.mkdir(exist_ok=True)
    
    output_csv_path = output_directory / 'geometric_features.csv'

    # Processa as anotações
    features_df = process_yolo_annotations(str(labels_directory))

    if not features_df.empty:
        # Salva o DataFrame em um arquivo CSV
        features_df.to_csv(output_csv_path, index=False)
        print("-" * 50)
        print(f"Processamento concluído com sucesso!")
        print(f"DataFrame com {features_df.shape[0]} amostras e {features_df.shape[1]} colunas.")
        print(f"Arquivo de features salvo em: {output_csv_path}")
        print("Amostra do resultado:")
        print(features_df.head())
        print("-" * 50)
    else:
        print("Nenhuma feature foi extraída. O arquivo CSV não foi gerado.")


if __name__ == '__main__':
    main()
