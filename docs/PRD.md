# PRD - Identificação de Bovinos via Pose Estimation e Machine Learning

## 1. Visão Geral e Objetivo

O objetivo deste projeto é desenvolver um sistema de Visão Computacional e Machine Learning capaz de identificar individualmente vacas leiteiras a partir de vídeos/imagens em vista superior (top-view). O sistema utiliza um modelo de detecção de pontos-chave (YOLO-Pose) para localizar partes anatômicas do animal e, a partir da geometria desses pontos, um modelo classificador (XGBoost) prevê a identidade do animal.

## 2. Status Atual do Projeto (Baseline)

O projeto já superou a fase inicial de preparação de dados e treinamento base da visão computacional.

- **Ambiente de Treinamento:** Google Colab Pro, consumindo dados diretamente do Google Drive.
- **Dataset:** Imagens já separadas em conjuntos de `train`, `val` e `test`.
- **Anotações:** Convertidas com sucesso para o padrão YOLO-Pose.
- **Modelos Base:** YOLO26x-pose (para keypoints) e YOLO26x.
- **Scripts Existentes:** Preparação de dataset, validação de anotações e treinamento com K-Fold Cross Validation já implementados na pasta `src/`.

## 3. Escopo da Próxima Fase

A próxima etapa foca na transição da detecção de coordenadas brutas (pixels) para a extração de inteligência (features biométricas) e classificação final. O escopo compreende:

1. **Análise do Pipeline Atual:** Garantir que o modelo YOLO treinado (`best.pt`) está exportando as coordenadas (X, Y) e a confiança de cada um dos 8 keypoints com precisão suficiente.
2. **Engenharia de Features (Passo B):** Desenvolvimento de um script para receber as saídas do YOLO e calcular atributos geométricos (ângulos e distâncias relativas).
3. **Modelo de Classificação (Passo C):** Treinamento de um modelo XGBoost utilizando os atributos geométricos gerados para classificar o `cow_id`.
4. **Pipeline End-to-End (Inferência):** Criação de um script unificado que aceita uma imagem crua e retorna o ID da vaca previsto.

## 4. Requisitos Técnicos

### 4.1. Extração de Features Geométricas

O script de extração deve processar as coordenadas $(X_i, Y_i)$ dos 8 pontos-chave (Head, Neck, Withers, Back, Hook, Hip ridge, Tail head, Pin).

- **Cálculo de Distâncias:** Distância Euclidiana entre pontos conectados anatômicamente (ex: Head ao Neck, Withers ao Back).
- **Cálculo de Ângulos:** Para calcular o ângulo no vértice $B(x_2, y_2)$ formado pelos pontos adjacentes $A(x_1, y_1)$ e $C(x_3, y_3)$, o algoritmo deve utilizar:

$$\theta = \left| \text{degrees}(\arctan2(y_3 - y_2, x_3 - x_2) - \arctan2(y_1 - y_2, x_1 - x_2)) \right|$$

_O script deve normalizar os ângulos para garantir que estejam sempre entre 0 e 180 graus._

- **Tratamento de Pontos Faltantes:** Se o YOLO não detectar um ponto (confiança muito baixa), o script de feature extraction deve possuir uma estratégia de imputação (ex: média da vaca, se no treino) ou retornar erro/aviso de "análise impossível" na inferência.

### 4.2. Modelo de Classificação (XGBoost)

- **Entrada:** DataFrame tabular contendo as features geradas (ângulos e distâncias).
- **Variável Alvo (Target):** `cow_id`.
- **Treinamento:** Deve ser executado no Colab Pro. Os hiperparâmetros devem ser otimizados (ex: `max_depth`, `learning_rate`, `n_estimators`).
- **Métricas de Avaliação:** Acurácia, F1-Score (macro/micro, devido a possíveis desbalanceamentos) e Matriz de Confusão.

### 4.3. Pipeline End-to-End

O script final (`inference_pipeline.py`) deve executar sequencialmente:

1. Carrega a Imagem.
2. `yolo_model.predict(imagem)` -> Extrai lista de dicionários com (X, Y).
3. `extract_features(coordenadas)` -> Retorna array/DataFrame de 1 linha.
4. `xgboost_model.predict(features)` -> Retorna o `cow_id`.

## 5. Proposta de Nova Estrutura de Diretórios

Para acomodar a nova fase sem quebrar a organização atual, a seguinte estrutura deve ser adotada:

```text
├── dataset
│   └── train
├── models
│   ├── yolo26x-pose.pt
│   ├── yolo26x.pt
│   ├── best_yolo_cow.pt          <-- Novo: Pesos finos do YOLO treinado
│   └── xgboost_cow_id.pkl        <-- Novo: Modelo classificador salvo
├── README.md
├── requirements-conda.txt
├── requirements.txt
└── src
    ├── convert_labels_to_yolo_pose.py
    ├── key_point_detection_cow.py
    ├── prepare_dataset.py
    ├── train_yolo_kfold_colab.ipynb
    ├── train_yolo_kfold.py
    ├── validate_annotations.py
    ├── extract_geometric_features.py  <-- Novo: Passo B (Extração de ângulos/distâncias)
    ├── train_xgboost_classifier.py    <-- Novo: Passo C (Treino do ML tabular)
    └── inference_pipeline.py          <-- Novo: Pipeline End-to-End

```

## 6. Critérios de Aceite

1. **Script de Features:** `extract_geometric_features.py` gera um arquivo `.csv` tabular contendo colunas para cada ângulo/distância e a respectiva coluna `cow_id` a partir das anotações de treino/validação.
2. **Treinamento XGBoost:** `train_xgboost_classifier.py` lê o `.csv`, treina o modelo, exibe as métricas de validação e salva o arquivo `.pkl` ou `.json` do modelo final na pasta `models/`.
3. **Pipeline de Inferência:** O `inference_pipeline.py` processa com sucesso uma imagem de teste inédita, sem erros de tipagem entre a saída do PyTorch/Ultralytics e a entrada do XGBoost/Pandas, retornando o ID numérico/string da vaca.

---

Gostaria que eu começasse elaborando o código em Python para o arquivo `extract_geometric_features.py` usando Pandas e NumPy para calcular essas features geométricas?
