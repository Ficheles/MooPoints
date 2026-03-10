# Funcionamento do Projeto

Este documento descreve o fluxo técnico do projeto de ponta a ponta, desde a preparação dos dados até o uso da API e da interface Streamlit.

## 1) Arquitetura em alto nível

```mermaid
flowchart LR
    A[Imagens de entrada] --> B[YOLO Pose]
    B --> C[Keypoints]
    C --> D[Extração de features geométricas]
    D --> E[XGBoost]
    E --> F[Classe prevista / desconhecida]
```

## 2) Fluxo de treino (pipeline recomendado)

```mermaid
flowchart TD
    A[data/fotos_classificar] --> B[prepare_classification_dataset.py]
    B --> C[data/datasets/classifications]
    C --> D[extract_geometric_features.py]
    D --> E[geometric_features.csv]
    E --> F[train_xgboost_classifier.py]
    F --> G[models/xgboost/*.pkl + *.json]
```

### Etapas

1. Organizar as imagens por classe em `data/fotos_classificar`.
2. Gerar splits de treino/validação/teste sem vazamento por sessão.
3. Rodar inferência de keypoints com YOLO Pose.
4. Extrair distâncias, ângulos e áreas triangulares.
5. Treinar o classificador XGBoost e salvar artefatos.

## 3) Fluxo de inferência na API

```mermaid
sequenceDiagram
    participant U as Cliente
    participant API as FastAPI
    participant YOLO as YOLO Pose
    participant FEAT as Feature Builder
    participant XGB as XGBoost

    U->>API: POST /cows/classify (imagem)
    API->>YOLO: Detectar keypoints
    YOLO-->>API: Keypoints + confiança
    API->>FEAT: Construir vetor de features
    FEAT-->>API: Vetor numérico
    API->>XGB: Predição + probabilidade
    XGB-->>API: Classe e score
    API-->>U: recognized, predicted_class, confidence, reason
```

### Regras principais

- Se não houver keypoints detectados, a resposta retorna `recognized=false`.
- Se a confiança ficar abaixo do limiar, retorna `recognized=false` com motivo técnico.
- Se superar o limiar, retorna `recognized=true` com `predicted_class`.

## 4) Fluxo de uso com Streamlit

```mermaid
flowchart TD
    A[Usuário envia imagem na UI] --> B[Streamlit]
    B --> C[POST /cows/classify]
    C --> D[Resposta JSON da API]
    D --> E[Card de resultado na UI]
```

## 5) Módulos e responsabilidades

- `src/config/geometry.py`: fonte única de constantes geométricas.
- `src/utils/geometry.py`: funções geométricas base (distância, ângulo, área).
- `src/utils/keypoint_features.py`: montagem de features para inferência/classificação.
- `src/classification/extract_geometric_features.py`: extração em lote para treino.
- `src/classification/train_xgboost_classifier.py`: treino e persistência de artefatos.
- `src/api.py`: orquestra inferência e disponibiliza endpoints HTTP.
- `src/ui/streamlit_app.py`: experiência de usuário para classificação.

## 6) Comandos úteis

```bash
# API
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Streamlit
streamlit run src/ui/streamlit_app.py

# Extração de features
python -m src.classification.extract_geometric_features \
  --dataset-root data/datasets/classifications \
  --model-path models/yolo/best.pt \
  --output-csv data/datasets/classifications/geometric_features.csv

# Treino XGBoost
python -m src.classification.train_xgboost_classifier \
  --features-csv data/datasets/classifications/geometric_features.csv \
  --models-dir models/xgboost
```
