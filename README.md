# Cow Classifier

Projeto para preparação de dataset de bovinos, validação de anotações e teste de detecção/pose com YOLO.

Também inclui uma API FastAPI para cadastro, identificação, remoção e listagem de vacas em base SQLite.

## Visão geral

Este repositório possui três fluxos principais:

1. **Preparar dataset** com links simbólicos e criação de folds.
2. **Validar anotações** para checar bbox e keypoints obrigatórios.
3. **Rodar inferência de pose** com o script `key_point_detection_cow.py`.
4. **Converter labels para YOLO Pose** com o script `convert_labels_to_yolo_pose.py`.
5. **Treinar com K-Fold + transfer learning** com o script `train_yolo_kfold.py`.
6. **Executar API FastAPI** para cadastro/identificação em SQLite.

## Estrutura principal

- `src/prepare_dataset.py`: organiza arquivos, cria links em `fotos_anotadas/00_dataset` e gera folds em `dataset/` mantendo labels originais em `.json`.
- `src/validate_annotations.py`: valida as anotações no diretório preparado.
- `src/key_point_detection_cow.py`: executa detecção + pose usando modelos YOLO.
- `src/convert_labels_to_yolo_pose.py`: lê labels `.json` (Label Studio) e gera labels `.txt` no formato YOLO Pose.
- `src/train_yolo_kfold.py`: treina YOLO Pose usando os folds já prontos em `dataset/`.
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
python src/prepare_dataset.py
```

### 2) Validar anotações

```bash
python src/validate_annotations.py
```

### 3) Executar inferência de pose

```bash
python src/key_point_detection_cow.py
```

### 4) Converter labels para YOLO Pose (obrigatório antes do treino)

```bash
python src/convert_labels_to_yolo_pose.py --dataset-root /app/dataset
```

### 5) Treinar com K-Fold (transfer learning)

```bash
python src/train_yolo_kfold.py \
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
python src/train_yolo_kfold.py --continue-from-best
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
- `DELETE /cows/{cow_id}`
	- Ação: remove cadastro da vaca e sua imagem
- `GET /cows`
	- Ação: lista vacas cadastradas com `id` e imagem de cadastro (`image_url`)
	- Query opcional: `include_base64=true` para retornar também a imagem em base64

Endpoint auxiliar para visualizar a imagem de cadastro:

- `GET /cows/{cow_id}/image`

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
python src/prepare_dataset.py
python src/validate_annotations.py
python src/key_point_detection_cow.py
python src/convert_labels_to_yolo_pose.py --dataset-root /app/dataset
python src/train_yolo_kfold.py --epochs 50 --continue-from-best
```

## Dicas rápidas

- Se aparecer aviso do Ultralytics sobre config, o projeto já usa `YOLO_CONFIG_DIR` no compose.
- Para limpar containers/parar execução:

```bash
docker compose down
```
