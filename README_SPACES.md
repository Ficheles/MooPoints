---
title: Cow Classifier
emoji: 🐮
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.55.0
app_file: app.py
pinned: false
license: mit
---

# 🐮 Cow Classifier

Sistema de identificação e classificação de vacas usando visão computacional.

## 🎯 Funcionalidades

- **Detecção de Keypoints**: YOLO Pose para detectar pontos-chave do corpo da vaca
- **Extração de Features**: Características geométricas baseadas em distâncias e ângulos
- **Classificação**: XGBoost para identificar vacas conhecidas
- **API REST**: FastAPI com endpoints de cadastro, identificação e classificação
- **Interface Web**: Streamlit para uso interativo

## 🏗️ Arquitetura

```
YOLO Pose → Features Geométricas → XGBoost Classifier
     ↓              ↓                      ↓
  Keypoints    Distâncias/Ângulos    Identificação
```

## 🔗 Endpoints da API

- `POST /cows/register` - Cadastrar nova vaca
- `POST /cows/identify` - Identificar vaca por similaridade
- `POST /cows/classify` - Classificar vaca com XGBoost
- `GET /cows` - Listar vacas cadastradas
- `DELETE /cows/{id}` - Remover cadastro

## 📚 Repositório

GitHub: [Ficheles/MooPoints](https://github.com/Ficheles/MooPoints)

## 🛠️ Tecnologias

- **YOLOv11 Pose** - Detecção de keypoints
- **XGBoost** - Classificação
- **FastAPI** - API REST
- **Streamlit** - Interface Web
- **scikit-learn** - Processamento de features
