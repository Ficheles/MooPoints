"""
Módulo para Treinamento de Classificador XGBoost.

Este script lê o arquivo de features geométricas gerado pelo
extract_geometric_features.py, treina um modelo XGBoost para classificar
a identidade das vacas (cow_id) e salva o modelo treinado.

Autor: GitHub Copilot
Data: 2026-03-05
"""
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_classifier():
    """
    Função principal para carregar os dados, treinar, avaliar e salvar o modelo.
    """
    project_root = Path(__file__).parent.parent
    features_path = project_root / 'dataset' / 'data' / 'geometric_features.csv'
    model_output_path = project_root / 'models' / 'xgboost_cow_id.pkl'
    
    model_output_path.parent.mkdir(exist_ok=True)

    print("Iniciando o treinamento do classificador XGBoost...")
    print(f"Carregando features de: {features_path}")

    # 1. Carregamento de Dados
    df = pd.read_csv(features_path)

    # 2. Pré-processamento
    print("Realizando pré-processamento dos dados...")
    
    # Separa features (X) e alvo (y)
    X = df.drop(columns=['filename', 'cow_id'])
    y = df['cow_id']

    # Codifica o alvo (cow_id) para formato numérico
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Tratamento de valores ausentes (NaN)
    # Usamos a média para preencher os pontos faltantes
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # 3. Divisão em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

    # 4. Treinamento do Modelo
    print("Treinando o modelo XGBClassifier...")
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # 5. Avaliação do Modelo
    print("Avaliando o desempenho do modelo...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print("-" * 50)
    print("Métricas de Avaliação no Conjunto de Teste:")
    print(f"  - Acurácia: {accuracy:.4f}")
    print(f"  - F1-Score (Macro): {f1:.4f}")
    print("-" * 50)

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    confusion_matrix_path = project_root / 'docs' / 'confusion_matrix.png'
    plt.savefig(confusion_matrix_path)
    print(f"Matriz de confusão salva em: {confusion_matrix_path}")

    # 6. Salvando o Modelo
    print(f"Salvando o modelo treinado em: {model_output_path}")
    joblib.dump(model, model_output_path)
    
    # Salvar também o LabelEncoder para decodificar as predições na inferência
    label_encoder_path = project_root / 'models' / 'label_encoder.pkl'
    joblib.dump(le, label_encoder_path)
    print(f"Label encoder salvo em: {label_encoder_path}")
    print("-" * 50)
    print("Treinamento concluído com sucesso!")


if __name__ == '__main__':
    train_classifier()
