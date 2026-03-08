import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Treina XGBoost com features geométricas para 30 classes.")
    parser.add_argument(
        "--features-csv",
        default="dataset_classification/geometric_features.csv",
        help="CSV com features geométricas e colunas class_name/split.",
    )
    parser.add_argument("--models-dir", default="models", help="Diretório de saída dos artefatos do modelo.")
    parser.add_argument(
        "--unknown-threshold",
        type=float,
        default=-1.0,
        help="Limiar fixo de confiança para classe desconhecida. Se <0, calibra no split val.",
    )
    return parser.parse_args()


def split_dataset(df: pd.DataFrame):
    if "split" not in df.columns:
        raise ValueError("CSV de features precisa da coluna 'split' com train/val/test.")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty:
        raise ValueError("Split train vazio no CSV de features.")

    return train_df, val_df, test_df


def feature_columns(df: pd.DataFrame):
    excluded = {"image_path", "class_name", "split"}
    return [col for col in df.columns if col not in excluded]


def evaluate(model, label_encoder, imputer, feature_cols, eval_df: pd.DataFrame):
    if eval_df.empty:
        return None

    x_eval = imputer.transform(eval_df[feature_cols])
    y_eval = label_encoder.transform(eval_df["class_name"])

    pred = model.predict(x_eval)
    proba = model.predict_proba(x_eval)
    max_proba = proba.max(axis=1)

    return {
        "samples": int(len(eval_df)),
        "accuracy": float(accuracy_score(y_eval, pred)),
        "f1_macro": float(f1_score(y_eval, pred, average="macro")),
        "max_proba_mean": float(np.mean(max_proba)),
        "max_proba_p10": float(np.percentile(max_proba, 10)),
    }


def calibrate_threshold(model, label_encoder, imputer, feature_cols, val_df: pd.DataFrame, fixed_threshold: float):
    if fixed_threshold >= 0:
        return float(fixed_threshold)

    if val_df.empty:
        return 0.55

    x_val = imputer.transform(val_df[feature_cols])
    y_val = label_encoder.transform(val_df["class_name"])

    pred = model.predict(x_val)
    proba = model.predict_proba(x_val)
    max_proba = proba.max(axis=1)

    correct_mask = pred == y_val
    correct_scores = max_proba[correct_mask]

    if len(correct_scores) == 0:
        return 0.55

    threshold = float(np.percentile(correct_scores, 10))
    return float(min(0.95, max(0.35, threshold)))


def main():
    args = parse_args()
    features_csv = Path(args.features_csv)
    models_dir = Path(args.models_dir)

    if not features_csv.exists():
        raise FileNotFoundError(f"Arquivo de features não encontrado: {features_csv}")

    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)
    if "class_name" not in df.columns:
        raise ValueError("CSV de features precisa da coluna 'class_name'.")

    train_df, val_df, test_df = split_dataset(df)
    feat_cols = feature_columns(df)

    x_train = train_df[feat_cols]
    y_train_raw = train_df["class_name"]

    imputer = SimpleImputer(strategy="median")
    x_train_imp = imputer.fit_transform(x_train)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(x_train_imp, y_train)

    threshold = calibrate_threshold(model, label_encoder, imputer, feat_cols, val_df, args.unknown_threshold)

    train_metrics = evaluate(model, label_encoder, imputer, feat_cols, train_df)
    val_metrics = evaluate(model, label_encoder, imputer, feat_cols, val_df)
    test_metrics = evaluate(model, label_encoder, imputer, feat_cols, test_df)

    joblib.dump(model, models_dir / "xgboost_cow_id.pkl")
    joblib.dump(label_encoder, models_dir / "xgb_label_encoder.pkl")
    joblib.dump(imputer, models_dir / "xgb_imputer.pkl")

    (models_dir / "xgb_feature_columns.json").write_text(
        json.dumps({"feature_columns": feat_cols}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (models_dir / "xgb_unknown_threshold.json").write_text(
        json.dumps({"unknown_threshold": threshold}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "features_csv": str(features_csv.resolve()),
        "num_classes": int(len(label_encoder.classes_)),
        "classes": label_encoder.classes_.tolist(),
        "unknown_threshold": threshold,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "artifacts": {
            "model": str((models_dir / "xgboost_cow_id.pkl").resolve()),
            "label_encoder": str((models_dir / "xgb_label_encoder.pkl").resolve()),
            "imputer": str((models_dir / "xgb_imputer.pkl").resolve()),
            "feature_columns": str((models_dir / "xgb_feature_columns.json").resolve()),
            "threshold": str((models_dir / "xgb_unknown_threshold.json").resolve()),
        },
    }

    report_path = models_dir / "xgb_training_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("Treino XGBoost concluído")
    print(f"- classes: {report['num_classes']}")
    print(f"- limiar desconhecida: {threshold:.4f}")
    print(f"- relatório: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
