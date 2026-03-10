# Cow Classifier

Projeto para preparação de dataset de bovinos, validação de anotações e teste de detecção/pose com YOLO.

Também inclui uma API FastAPI para:

- cadastro/listagem de vacas em base SQLite (fluxo de similaridade), e
- classificação por 30 classes com XGBoost (fluxo principal de produção).

## Visão geral

Este repositório possui fluxos principais:

1. **Preparar dataset** com links simbólicos e criação de folds.
2. **Validar anotações** para checar bbox e keypoints obrigatórios.
3. **Rodar inferência de pose** com o script `key_point_detection_cow.py`.
4. **Converter labels para YOLO Pose** com o script `convert_labels_to_yolo_pose.py`.
5. **Treinar com K-Fold + transfer learning** com o script `train_yolo_kfold.py`.
6. **Executar API FastAPI** para cadastro/identificação em SQLite e classificação via XGBoost.
7. **Classificação por 30 classes com features geométricas** (YOLO pose → features → XGBoost).

## Estrutura principal

- `src/keypoints/prepare_dataset.py`: organiza arquivos, cria links em `fotos_anotadas/00_dataset` e gera folds em `dataset/` mantendo labels originais em `.json`.
- `src/keypoints/validate_annotations.py`: valida as anotações no diretório preparado.
- `src/keypoints/key_point_detection_cow.py`: executa detecção + pose usando modelos YOLO.
- `src/keypoints/convert_labels_to_yolo_pose.py`: lê labels `.json` (Label Studio) e gera labels `.txt` no formato YOLO Pose.
- `src/keypoints/train_yolo_kfold.py`: treina YOLO Pose usando os folds já prontos em `dataset/`.
- `src/classification/prepare_classification_dataset.py`: organiza `fotos_classificar` em `dataset_classification` com teste estratificado (10%) e 5 folds train/val sem vazamento por `session_id`.
- `src/classification/generate_geometric_features_from_dataset.py`: usa modelo YOLO pose para gerar automaticamente keypoints/features geométricas a partir de `dataset_classification`.
- `src/classification/train_xgboost_classifier.py`: treina classificador XGBoost com as features, salva artefatos e limiar de desconhecida.
- `src/classification/train_image_classifier_kfold.py`: treina classificador por imagem em `dataset_classification/fold_*`.
- `src/classification/evaluate_image_classifier.py`: avalia checkpoint treinado no conjunto `dataset_classification/test`.
- `src/classification/inference_image_classifier.py`: classifica uma imagem em uma das classes treinadas.
- `docker-compose.yml`: execução do projeto em container.

## Como usar localmente (Python)

### Pré-requisitos

- Python 3.12+
- Pip

### Instalação

```bash
pip install -r requirements.txt
```

### 1) Preparar dataset

```bash
python src/keypoints/prepare_dataset.py
```

### 2) Validar anotações

```bash
python src/keypoints/validate_annotations.py
```

### 3) Executar inferência de pose

```bash
python src/keypoints/key_point_detection_cow.py
```

### 3.1) Predizer keypoints de uma imagem com `best.pt`

Use o script de predição por imagem informando o caminho da imagem:

```bash
python -m src.keypoints.predict_keypoints_from_image \
	--image-path /caminho/para/imagem.jpg \
	--model-path models/yolo/best.pt
```

Para salvar também JSON e imagem anotada:

```bash
python -m src.keypoints.predict_keypoints_from_image \
	--image-path /caminho/para/imagem.jpg \
	--model-path models/yolo/best.pt \
	--save-json outputs/keypoints.json \
	--save-image outputs/keypoints_annotated.jpg
```

### 4) Converter labels para YOLO Pose (obrigatório antes do treino)

```bash
python src/keypoints/convert_labels_to_yolo_pose.py --dataset-root /app/dataset
```

### 5) Treinar com K-Fold (transfer learning)

```bash
python src/keypoints/train_yolo_kfold.py \
	--dataset-root /app/dataset \
	--models-dir /app/models \
	--base-model yolo11x-pose.pt \
	--epochs 100 \
	--batch 8 \
	--imgsz 640 \
	--device cpu
```

Para continuar o treino depois usando o melhor checkpoint salvo de cada fold:

```bash
python src/keypoints/train_yolo_kfold.py --continue-from-best
```

O relatório consolidado é salvo em:

- `/app/runs/kfold_pose/kfold_metrics_report.json`

### 6) Executar API FastAPI

Suba a API:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints principais:

- `POST /cows/register`
  - Entrada: `multipart/form-data` com campo `image`
  - Ação: detecta keypoints da vaca, extrai features geométricas e salva no SQLite (`data/cows.db`)
- `POST /cows/identify`
  - Entrada: `multipart/form-data` com campo `image`
  - Query opcional: `similarity_threshold` (padrão: `0.98`)
  - Ação: compara features da imagem enviada com a base SQLite e informa se foi reconhecida
- `POST /cows/classify`
	- Entrada: `multipart/form-data` com campo `image`
	- Query opcional: `confidence_threshold` (se ausente, usa limiar salvo no treino XGBoost)
	- Ação: extrai keypoints com YOLO pose, gera features geométricas e classifica com XGBoost
	- Retorno esperado:
		- `recognized=true` + `predicted_class` quando pertence a uma das classes treinadas
		- `recognized=false` quando não pertence (abaixo do limiar)
		- `reason` com motivo técnico (`recognized`, `below_confidence_threshold`, `no_keypoints_detected`, `partial_keypoints_detected`)
- `DELETE /cows/{cow_id}`
  - Ação: remove cadastro da vaca e sua imagem
- `GET /cows`
  - Ação: lista vacas cadastradas com `id` e imagem de cadastro (`image_url`)
  - Query opcional: `include_base64=true` para retornar também a imagem em base64

Endpoint auxiliar para visualizar a imagem de cadastro:

- `GET /cows/{cow_id}/image`

### 6.1) Interface Web (Streamlit)

Com a API em execução, suba a interface web:

```bash
streamlit run src/ui/streamlit_app.py
```

Fluxo da interface:

- usuário envia uma foto da vaca;
- a UI chama `POST /cows/classify` com limiar configurável;
- retorno exibido como:
	- **Reconhecida**: mostra `predicted_class`;
	- **Desconhecida**: quando `recognized=false`;
	- mensagem amigável baseada em `reason`.

Recursos de UX da tela:

- pré-visualização da imagem enviada;
- status visual de processamento;
- cartão de resultado com confiança e limiar usado;
- teste rápido de conexão com a API pelo endpoint `GET /cows`.

### 7) Classificação por 30 classes (fluxo fim a fim recomendado)

Este é o fluxo principal para produção: **YOLO pose → features geométricas → XGBoost → API/Streamlit**.

#### Estrutura esperada de entrada (`fotos_classificar`)

Formato recomendado (uma pasta por vaca/classe):

```text
fotos_classificar/
├── cow_01/
│   ├── 20260101_040807_baia16_IPC1_001.jpg
│   └── ...
├── cow_02/
│   └── ...
...
└── cow_30/
		└── ...
```

#### 7.1 Gerar `dataset_classification` com split sem vazamento por sessão

```bash
python src/classification/prepare_classification_dataset.py \
	--input-root fotos_classificar \
	--output-root dataset_classification \
	--test-size 0.10 \
	--n-splits 5 \
	--clean-output
```

Resultado:

```text
dataset_classification/
├── classes.csv
├── splits_manifest.csv
├── test/
│   ├── cow_01/*.jpg
│   ├── ...
│   └── cow_30/*.jpg
├── fold_0/
│   ├── train/cow_01/*.jpg ... cow_30/*.jpg
│   └── val/cow_01/*.jpg ... cow_30/*.jpg
...
└── fold_4/
		├── train/cow_01/*.jpg ... cow_30/*.jpg
		└── val/cow_01/*.jpg ... cow_30/*.jpg
```

Observações:

- O teste usa 10% estratificado por classe e sem misturar `session_id` com os folds de desenvolvimento.
- Os 90% restantes são divididos em 5 folds (`train`/`val`) também sem vazamento por `session_id`.

#### 7.2 Gerar features geométricas automaticamente com o modelo de pose

Use o `best.pt` de pose já treinado para anotar implicitamente as imagens (inferência de keypoints) e gerar o CSV de features:

```bash
python src/classification/generate_geometric_features_from_dataset.py \
	--dataset-root dataset_classification \
	--fold 0 \
	--model-path models/yolo11x-pose.pt \
	--output-csv dataset_classification/geometric_features.csv
```

Saída:

- `dataset_classification/geometric_features.csv`

#### 7.3 Treinar XGBoost com as features

```bash
python src/classification/train_xgboost_classifier.py \
	--features-csv dataset_classification/geometric_features.csv \
	--models-dir models
```

Artefatos gerados em `models/`:

- `xgboost_cow_id.pkl`
- `xgb_label_encoder.pkl`
- `xgb_imputer.pkl`
- `xgb_feature_columns.json`
- `xgb_unknown_threshold.json`
- `xgb_training_report.json`

#### 7.4 Subir API e validar classificação

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Teste rápido do endpoint de classificação:

```bash
curl -X POST "http://localhost:8000/cows/classify" \
	-F "image=@/caminho/para/imagem.jpg"
```

#### 7.5 Usar Streamlit no fluxo final

```bash
streamlit run src/ui/streamlit_app.py
```

Comportamento esperado:

- mostra classe prevista quando `recognized=true`;
- mostra desconhecida quando `recognized=false`;
- exibe mensagem amigável baseada em `reason`.

---

### Fluxo alternativo (legado): classificação por imagem direta

Os scripts abaixo continuam disponíveis para o fluxo de classificador de imagem (sem features geométricas):

- `src/classification/train_image_classifier_kfold.py`
- `src/classification/evaluate_image_classifier.py`
- `src/classification/inference_image_classifier.py`

#### Treinar classificador K-Fold (legado)

```bash
python src/classification/train_image_classifier_kfold.py \
	--dataset-root dataset_classification \
	--models-dir models \
	--base-model yolo11n-cls.pt \
	--epochs 50 \
	--batch 16 \
	--imgsz 640 \
	--device cpu
```

Relatório consolidado:

- `runs/kfold_classification/kfold_classification_report.json`

#### Inferência de uma imagem

```bash
python src/classification/inference_image_classifier.py \
	/caminho/para/imagem.jpg \
	--model-path runs/kfold_classification/cls_fold_0/weights/best.pt
```

#### Avaliar no teste estratificado (10%)

```bash
python src/classification/evaluate_image_classifier.py \
	--test-root dataset_classification/test \
	--model-path runs/kfold_classification/cls_fold_0/weights/best.pt \
	--output-json runs/kfold_classification/test_metrics_fold0.json
```

## Sobre o script `key_point_detection_cow.py`

Esse script:

- Carrega uma imagem de entrada (`/app/dataset_samples/person.png`).
- Usa dois modelos:
  - detecção (`yolo26n.pt`)
  - pose (`yolo26n-pose.pt`)
- Desenha caixa e esqueleto dos keypoints.
- Salva a saída em `/app/dataset_samples/person-pose_with_skeleton.png`.

Os nomes dos modelos são configuráveis por variáveis de ambiente:

- `MODEL_DIR` (padrão `/app/models`)
- `DETECTION_MODEL_NAME` (padrão `yolo26n.pt`)
- `POSE_MODEL_NAME` (padrão `yolo26n-pose.pt`)

## Sobre o script `train_yolo_kfold.py`

Esse script foi feito para treinar em cima do dataset já separado por folds (`dataset/fold_*`) com `images/` e `labels/`.

> Antes do treino, execute `convert_labels_to_yolo_pose.py` para transformar os `.json` em `.txt` no formato esperado pelo YOLO.

O script de conversão:

- lê os labels originais `fold_*/labels/*/*.json`;
- gera os correspondentes `fold_*/labels/*/*.txt`;

Ele faz automaticamente:

- Busca dos `data_fold_*.yaml` em cada fold.
- Treino por fold com transfer learning a partir de um modelo base em `/app/models`.
- Registro das métricas por fold (box e pose).
- Salvamento do melhor e último checkpoint (`best.pt` e `last.pt`).
- Geração de relatório consolidado com estatísticas.

Principais variáveis/opções:

- `MODEL_DIR` (padrão `/app/models`)
- `TRAIN_MODEL_NAME` (padrão `yolo26n-pose.pt`)
- `DATASET_ROOT` (padrão `/app/dataset`)
- `CONTINUE_FROM_BEST=true` para retomar do `best.pt`

Template de métricas geradas pelo relatório (estrutura):

```text
k_folds: 5
Box_mAP50 (média dos folds): ...
Box_mAP50-95 (média dos folds): ...
Pose_mAP50 (média dos folds): ...
Pose_mAP50-95 (média dos folds): ...
Pose_mAP50-95 (melhor fold): ...

Métricas no teste final:
accuracy: ...
f1_macro: ...
top1_accuracy: ...
top3_accuracy: ...
top5_accuracy: ...

Com rejeição (confianca_min=0.30):
cobertura: ...
accuracy_aceitas: ...
f1_macro_aceitas: ...
```

## Como usar com Docker

### Pré-requisitos

- Docker
- Docker Compose

### 1) Garanta os modelos no host

Coloque os arquivos de modelo em `./models`:

- `./models/yolo26n.pt`
- `./models/yolo26n-pose.pt`

> O diretório `models` é montado como bind mount em `/app/models` e usado apenas em runtime.

### 2) Subir com compose

```bash
docker compose up --build
```

### 3) Executar comandos manuais (interativo)

```bash
docker compose run --rm prepare-dataset sh
```

Dentro do container, por exemplo:

```bash
python src/keypoints/prepare_dataset.py
python src/keypoints/validate_annotations.py
python src/keypoints/key_point_detection_cow.py
python src/keypoints/convert_labels_to_yolo_pose.py --dataset-root /app/dataset
python src/keypoints/train_yolo_kfold.py --epochs 50 --continue-from-best
```

## Dicas rápidas

- Se aparecer aviso do Ultralytics sobre config, o projeto já usa `YOLO_CONFIG_DIR` no compose.
- Para limpar containers/parar execução:

```bash
docker compose down
```
