import os
import yaml
import shutil
import json
import urllib.parse
from sklearn.model_selection import KFold, train_test_split

PASTA_ORIGINAL = "data/fotos_anotadas"
BASE_DIR = "data/datasets/keypoints"
DATASET_DIR = "00_dataset"
PASTA_LINKS = os.path.join(PASTA_ORIGINAL, DATASET_DIR)

K = 5
TEST_SIZE = 0.1


def criar_link_simbolico(origem, destino):
    if os.path.lexists(destino):
        os.remove(destino)
    os.symlink(os.path.abspath(origem), destino)

def organizar_arquivos():
    print("Iniciando organização dos arquivos...", flush=True)
    pares_gerados = 0
    os.makedirs(PASTA_LINKS, exist_ok=True)
    if os.path.exists(PASTA_ORIGINAL):
        subdirs = [
            d for d in os.listdir(PASTA_ORIGINAL)
            if os.path.isdir(os.path.join(PASTA_ORIGINAL, d)) and d != DATASET_DIR
        ]
        print(f"Subdiretórios encontrados: {subdirs}", flush=True)
    else:
        print(f"ERRO: Pasta {PASTA_ORIGINAL} não encontrada!", flush=True)
        return
    for subdir in subdirs:
        subdir_path = os.path.join(PASTA_ORIGINAL, subdir)
        key_points_path = os.path.join(subdir_path, "Key_points")
        if not os.path.exists(key_points_path):
            print(f"PULANDO {subdir}: Key_points não encontrada", flush=True)
            continue
        anotacao_files = os.listdir(key_points_path)
        print(f"Processando {subdir}: {len(anotacao_files)} arquivos encontrados em Key_points", flush=True)
        for arq_anotacao in anotacao_files:
            caminho_anotacao = os.path.join(key_points_path, arq_anotacao)
            if os.path.isdir(caminho_anotacao):
                continue
            try:
                with open(caminho_anotacao, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'task' in data and 'data' in data['task'] and 'img' in data['task']['data']:
                    img_path_encoded = data['task']['data']['img']
                    img_path_decoded = urllib.parse.unquote(img_path_encoded)
                    img_filename_raw = img_path_decoded.replace('\\', '/').split('/')[-1]
                    partes_nome = img_filename_raw.split('-', 1)
                    if len(partes_nome) == 2 and len(partes_nome[0]) == 8:
                        img_filename = partes_nome[1]
                    else:
                        img_filename = img_filename_raw
                    caminho_imagem_origem = os.path.join(subdir_path, img_filename)
                    print(f"  Tentando imagem: {caminho_imagem_origem}", flush=True)
                    if os.path.exists(caminho_imagem_origem):
                        nome_base_original = os.path.splitext(img_filename)[0]
                        novo_caminho_img = os.path.join(PASTA_LINKS, img_filename)
                        novo_caminho_json = os.path.join(PASTA_LINKS, f"{nome_base_original}.json")
                        criar_link_simbolico(caminho_imagem_origem, novo_caminho_img)
                        criar_link_simbolico(caminho_anotacao, novo_caminho_json)
                        print(f"Processado: {subdir}/{img_filename} -> {img_filename} / {nome_base_original}.json (symlink)", flush=True)
                        pares_gerados += 1
                    else:
                        print(f"AVISO: Imagem {img_filename} não encontrada em {subdir_path}", flush=True)
            except Exception as e:
                print(f"ERRO ao processar {caminho_anotacao}: {e}")

    print(f"Organização concluída. {pares_gerados} pares de links gerados em {PASTA_LINKS}.")

organizar_arquivos()

files = [f.split('.')[0] for f in os.listdir(PASTA_LINKS) if f.endswith('.jpg')]
files = sorted(files)
kf = KFold(n_splits=K, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(files)):
    fold_path = os.path.join(BASE_DIR, f"fold_{fold}")
    if os.path.exists(fold_path):
        shutil.rmtree(fold_path)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(fold_path, "images", split), exist_ok=True)
        os.makedirs(os.path.join(fold_path, "labels", split), exist_ok=True)

    train_idx_list = list(train_idx)
    test_idx = []
    if len(train_idx_list) > 1 and TEST_SIZE > 0:
        train_idx_list, test_idx = train_test_split(
            train_idx_list,
            test_size=TEST_SIZE,
            random_state=42,
            shuffle=True,
        )

    def mover_para_estrutura(indices, split_name):
        for i in indices:
            name = files[i]
            src_img = os.path.join(PASTA_LINKS, f"{name}.jpg")
            src_json = os.path.join(PASTA_LINKS, f"{name}.json")
            if os.path.exists(src_img) and os.path.exists(src_json):
                shutil.copy(src_img, f"{fold_path}/images/{split_name}/")
                shutil.copy(src_json, f"{fold_path}/labels/{split_name}/")

    mover_para_estrutura(train_idx_list, "train")
    mover_para_estrutura(val_idx, "val")
    mover_para_estrutura(test_idx, "test")

    config = {
        'path': os.path.abspath(fold_path),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'kpt_shape': [8, 3],
        'names': {0: 'cow'}
    }
    with open(os.path.join(fold_path, f"data_fold_{fold}.yaml"), 'w') as f:
        yaml.dump(config, f)
print(f"Estrutura criada em /{BASE_DIR} com {K} folds e {TEST_SIZE * 100:.0f}% de teste por fold.")