"""
API FastAPI para cadastro e identificação de vacas por features geométricas.

Endpoints:
- POST /cows/register: cadastra uma vaca com base em imagem
- POST /cows/identify: identifica se a vaca já existe na base
- DELETE /cows/{cow_id}: remove cadastro
- GET /cows: lista vacas cadastradas
"""

from __future__ import annotations

import base64
import json
import math
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ultralytics import YOLO

from src.classification.inference_pipeline import (
    ANGLE_TRIPLETS,
    DISTANCE_PAIRS,
    KEYPOINT_MAP,
    build_feature_dict,
    calculate_angle,
    calculate_distance,
)


PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolo" / "yolo11x-pose.pt"
DB_PATH = PROJECT_ROOT / "data" / "cows.db"
IMAGES_DIR = PROJECT_ROOT / "data" / "registered_cows"
XGB_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost" / "xgboost_cow_id.pkl"
XGB_ENCODER_PATH = PROJECT_ROOT / "models" / "xgboost" / "xgb_label_encoder.pkl"
XGB_IMPUTER_PATH = PROJECT_ROOT / "models" / "xgboost" / "xgb_imputer.pkl"
XGB_FEATURES_PATH = PROJECT_ROOT / "models" / "xgboost" / "xgb_feature_columns.json"
XGB_THRESHOLD_PATH = PROJECT_ROOT / "models" / "xgboost" / "xgb_unknown_threshold.json"


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _ensure_dirs() -> None:
    global DB_PATH, IMAGES_DIR

    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fallback_root = PROJECT_ROOT / ".runtime_data"
        DB_PATH = fallback_root / "cows.db"
        IMAGES_DIR = fallback_root / "registered_cows"
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    _ensure_dirs()
    conn = _get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                image_mime TEXT NOT NULL,
                keypoints_json TEXT NOT NULL,
                features_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _feature_columns() -> list[str]:
    cols: list[str] = []
    for p1, p2 in DISTANCE_PAIRS:
        cols.append(f"dist_{p1}_{p2}")
    for p1, p2, p3 in ANGLE_TRIPLETS:
        cols.append(f"angle_{p1}_{p2}_{p3}")
    return cols


FEATURE_COLUMNS = _feature_columns()


def _vector_from_features(features: dict[str, float]) -> np.ndarray:
    values = [float(features[col]) for col in FEATURE_COLUMNS]
    return np.array(values, dtype=float)


def _xgb_vector_from_features(features: dict[str, float], feature_columns: list[str]) -> np.ndarray:
    values = [float(features[col]) for col in feature_columns]
    return np.array(values, dtype=float)


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    sim = float(np.dot(v1, v2) / (n1 * n2))
    if math.isnan(sim):
        return 0.0
    return sim


def _to_jsonable_keypoints(keypoints: np.ndarray) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in keypoints]


class CowFeatureExtractor:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo YOLO não encontrado: {model_path}")
        self.model = YOLO(str(model_path))
        self._conf_schedule = (0.25, 0.15, 0.08, 0.05)
        self._imgsz_schedule = (640, 960, 1280)

    def _extract_keypoints_with_retries(self, image_bgr: np.ndarray) -> np.ndarray:
        best_partial_count = 0

        for imgsz in self._imgsz_schedule:
            for conf in self._conf_schedule:
                results = self.model.predict(
                    source=image_bgr,
                    task="pose",
                    conf=conf,
                    imgsz=imgsz,
                    verbose=False,
                )
                if not results:
                    continue

                keypoints_obj = getattr(results[0], "keypoints", None)
                if keypoints_obj is None or keypoints_obj.xy is None:
                    continue

                keypoints_batch = keypoints_obj.xy.cpu().numpy()
                if len(keypoints_batch) == 0:
                    continue

                keypoints = keypoints_batch[0]
                if keypoints.shape[0] >= 8:
                    return keypoints

                best_partial_count = max(best_partial_count, int(keypoints.shape[0]))

        if best_partial_count > 0:
            raise ValueError(
                f"Detecção parcial: encontrados {best_partial_count} pontos-chave, mas são necessários 8."
            )

        raise ValueError("Nenhuma vaca ou ponto-chave detectado na imagem.")

    def extract_from_image_array(self, image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        keypoints = self._extract_keypoints_with_retries(image_bgr)
        required_kpts = len(KEYPOINT_MAP)

        if keypoints.shape[0] < required_kpts:
            raise ValueError(
                f"Detecção parcial: encontrados {int(keypoints.shape[0])} pontos-chave, mas são necessários {required_kpts}."
            )

        if keypoints.shape[0] > required_kpts:
            keypoints = keypoints[:required_kpts]

        features = build_feature_dict(keypoints)

        return keypoints, features


class RegisterResponse(BaseModel):
    id: int
    message: str
    created_at: str


class IdentifyResponse(BaseModel):
    recognized: bool
    matched_id: int | None
    similarity: float | None
    threshold: float
    reason: str | None = None


class ClassifyResponse(BaseModel):
    recognized: bool
    predicted_class: str | None
    confidence: float | None
    threshold: float
    reason: str | None = None
    keypoints: list[list[float]] | None = None
    keypoint_names: list[str] | None = None


app = FastAPI(title="Cow Classifier API", version="1.0.0")
extractor: CowFeatureExtractor | None = None
xgb_model = None
xgb_encoder = None
xgb_imputer = None
xgb_feature_columns: list[str] | None = None
xgb_unknown_threshold: float = 0.55


def _load_xgb_artifacts() -> None:
    global xgb_model, xgb_encoder, xgb_imputer, xgb_feature_columns, xgb_unknown_threshold

    required = [XGB_MODEL_PATH, XGB_ENCODER_PATH, XGB_IMPUTER_PATH, XGB_FEATURES_PATH]
    if not all(path.exists() for path in required):
        xgb_model = None
        xgb_encoder = None
        xgb_imputer = None
        xgb_feature_columns = None
        xgb_unknown_threshold = 0.55
        return

    xgb_model = joblib.load(XGB_MODEL_PATH)
    xgb_encoder = joblib.load(XGB_ENCODER_PATH)
    xgb_imputer = joblib.load(XGB_IMPUTER_PATH)

    payload = json.loads(XGB_FEATURES_PATH.read_text(encoding="utf-8"))
    xgb_feature_columns = list(payload.get("feature_columns", []))

    if XGB_THRESHOLD_PATH.exists():
        threshold_payload = json.loads(XGB_THRESHOLD_PATH.read_text(encoding="utf-8"))
        xgb_unknown_threshold = float(threshold_payload.get("unknown_threshold", 0.55))
    else:
        xgb_unknown_threshold = 0.55


@app.on_event("startup")
def startup() -> None:
    global extractor
    _init_db()
    extractor = CowFeatureExtractor(MODEL_PATH)
    _load_xgb_artifacts()


def _validate_image(upload: UploadFile, raw_bytes: bytes) -> tuple[np.ndarray, str]:
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Arquivo de imagem vazio.")

    image_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Não foi possível decodificar a imagem enviada.")

    mime = upload.content_type or "image/jpeg"
    if not mime.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo enviado não é uma imagem válida.")

    return image_bgr, mime


@app.post("/cows/register", response_model=RegisterResponse)
async def register_cow(image: UploadFile = File(...)) -> RegisterResponse:
    if extractor is None:
        raise HTTPException(status_code=500, detail="Extrator de features não inicializado.")

    raw_bytes = await image.read()
    image_bgr, mime = _validate_image(image, raw_bytes)

    try:
        keypoints, features = extractor.extract_from_image_array(image_bgr)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    extension = Path(image.filename or "image.jpg").suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
        extension = ".jpg"

    file_name = f"cow_{uuid.uuid4().hex}{extension}"
    image_path = IMAGES_DIR / file_name
    image_path.write_bytes(raw_bytes)

    conn = _get_conn()
    try:
        cursor = conn.execute(
            """
            INSERT INTO cows (image_path, image_mime, keypoints_json, features_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(image_path),
                mime,
                json.dumps(_to_jsonable_keypoints(keypoints)),
                json.dumps(features),
                _now_iso(),
            ),
        )
        conn.commit()
        inserted_id = int(cursor.lastrowid)
    finally:
        conn.close()

    return RegisterResponse(
        id=inserted_id,
        message="Vaca cadastrada com sucesso.",
        created_at=_now_iso(),
    )


@app.post("/cows/identify", response_model=IdentifyResponse)
async def identify_cow(
    image: UploadFile = File(...),
    similarity_threshold: float = Query(default=0.98, ge=0.0, le=1.0),
) -> IdentifyResponse:
    if extractor is None:
        raise HTTPException(status_code=500, detail="Extrator de features não inicializado.")

    raw_bytes = await image.read()
    image_bgr, _ = _validate_image(image, raw_bytes)

    try:
        _, query_features = extractor.extract_from_image_array(image_bgr)
    except ValueError as exc:
        reason = "no_keypoints_detected"
        if "Detecção parcial" in str(exc):
            reason = "partial_keypoints_detected"
        return IdentifyResponse(
            recognized=False,
            matched_id=None,
            similarity=None,
            threshold=similarity_threshold,
            reason=reason,
        )

    query_vector = _vector_from_features(query_features)

    conn = _get_conn()
    try:
        rows = conn.execute("SELECT id, features_json FROM cows").fetchall()
    finally:
        conn.close()

    if not rows:
        return IdentifyResponse(
            recognized=False,
            matched_id=None,
            similarity=None,
            threshold=similarity_threshold,
            reason="empty_database",
        )

    best_id: int | None = None
    best_similarity = -1.0

    for row in rows:
        db_features = json.loads(row["features_json"])
        db_vector = _vector_from_features(db_features)
        similarity = _cosine_similarity(query_vector, db_vector)
        if similarity > best_similarity:
            best_similarity = similarity
            best_id = int(row["id"])

    recognized = best_similarity >= similarity_threshold
    reason = "recognized" if recognized else "below_similarity_threshold"

    return IdentifyResponse(
        recognized=recognized,
        matched_id=best_id if recognized else None,
        similarity=round(float(best_similarity), 6),
        threshold=similarity_threshold,
        reason=reason,
    )


@app.post("/cows/classify", response_model=ClassifyResponse)
async def classify_cow(
    image: UploadFile = File(...),
    confidence_threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    include_keypoints: bool = Query(default=False),
) -> ClassifyResponse:
    if extractor is None:
        raise HTTPException(status_code=500, detail="Extrator de features não inicializado.")

    if xgb_model is None or xgb_encoder is None or xgb_imputer is None or not xgb_feature_columns:
        raise HTTPException(status_code=500, detail="Modelo XGBoost não carregado.")

    effective_threshold = float(xgb_unknown_threshold if confidence_threshold is None else confidence_threshold)

    raw_bytes = await image.read()
    image_bgr, _ = _validate_image(image, raw_bytes)

    try:
        keypoints, features = extractor.extract_from_image_array(image_bgr)
    except ValueError as exc:
        reason = "no_keypoints_detected"
        if "Detecção parcial" in str(exc):
            reason = "partial_keypoints_detected"
        return ClassifyResponse(
            recognized=False,
            predicted_class=None,
            confidence=None,
            threshold=effective_threshold,
            reason=reason,
            keypoints=None,
            keypoint_names=None,
        )

    vector = _xgb_vector_from_features(features, xgb_feature_columns).reshape(1, -1)
    vector_imp = xgb_imputer.transform(vector)

    proba = xgb_model.predict_proba(vector_imp)[0]
    best_index = int(np.argmax(proba))
    best_confidence = float(proba[best_index])
    predicted_class = str(xgb_encoder.inverse_transform([best_index])[0])

    recognized = best_confidence >= effective_threshold
    reason = "recognized" if recognized else "below_confidence_threshold"

    return ClassifyResponse(
        recognized=recognized,
        predicted_class=predicted_class if recognized else None,
        confidence=round(best_confidence, 6),
        threshold=effective_threshold,
        reason=reason,
        keypoints=_to_jsonable_keypoints(keypoints) if include_keypoints else None,
        keypoint_names=[KEYPOINT_MAP[i] for i in range(len(KEYPOINT_MAP))] if include_keypoints else None,
    )


@app.delete("/cows/{cow_id}")
def delete_cow(cow_id: int) -> dict[str, str]:
    conn = _get_conn()
    try:
        row = conn.execute("SELECT image_path FROM cows WHERE id = ?", (cow_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Vaca não encontrada.")

        conn.execute("DELETE FROM cows WHERE id = ?", (cow_id,))
        conn.commit()
    finally:
        conn.close()

    image_path = Path(row["image_path"])
    if image_path.exists():
        image_path.unlink()

    return {"message": "Cadastro removido com sucesso."}


@app.get("/cows")
def list_cows(include_base64: bool = Query(default=False)) -> dict[str, list[dict[str, str | int | None]]]:
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT id, image_path, image_mime, created_at FROM cows ORDER BY id DESC"
        ).fetchall()
    finally:
        conn.close()

    data: list[dict[str, str | int | None]] = []
    for row in rows:
        image_path = Path(row["image_path"])
        item: dict[str, str | int | None] = {
            "id": int(row["id"]),
            "created_at": str(row["created_at"]),
            "image_url": f"/cows/{row['id']}/image",
        }
        if include_base64 and image_path.exists():
            raw = image_path.read_bytes()
            encoded = base64.b64encode(raw).decode("utf-8")
            item["image_base64"] = encoded
            item["image_mime"] = str(row["image_mime"])
        data.append(item)

    return {"items": data}


@app.get("/cows/{cow_id}/image")
def get_cow_image(cow_id: int) -> FileResponse:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT image_path, image_mime FROM cows WHERE id = ?",
            (cow_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail="Vaca não encontrada.")

    image_path = Path(row["image_path"])
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Imagem de cadastro não encontrada.")

    return FileResponse(path=image_path, media_type=row["image_mime"])
