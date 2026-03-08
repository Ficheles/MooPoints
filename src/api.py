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
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ultralytics import YOLO

from inference_pipeline import (
    ANGLE_TRIPLETS,
    DISTANCE_PAIRS,
    KEYPOINT_MAP,
    calculate_angle,
    calculate_distance,
)


PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolo11x-pose.pt"
DB_PATH = PROJECT_ROOT / "data" / "cows.db"
IMAGES_DIR = PROJECT_ROOT / "data" / "registered_cows"


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _ensure_dirs() -> None:
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

    def extract_from_image_array(self, image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        results = self.model(image_bgr, verbose=False)
        if not results:
            raise ValueError("Nenhum resultado retornado pelo modelo YOLO.")

        keypoints_batch = results[0].keypoints.xy.cpu().numpy()
        if len(keypoints_batch) == 0:
            raise ValueError("Nenhuma vaca ou ponto-chave detectado na imagem.")

        keypoints = keypoints_batch[0]
        if keypoints.shape[0] < 8:
            raise ValueError(f"Esperava 8 pontos-chave, mas recebeu {keypoints.shape[0]}.")

        keypoint_map = {KEYPOINT_MAP[i]: coord for i, coord in enumerate(keypoints)}

        features: dict[str, float] = {}
        for p1, p2 in DISTANCE_PAIRS:
            features[f"dist_{p1}_{p2}"] = float(calculate_distance(keypoint_map[p1], keypoint_map[p2]))

        for p1, p2, p3 in ANGLE_TRIPLETS:
            features[f"angle_{p1}_{p2}_{p3}"] = float(calculate_angle(keypoint_map[p1], keypoint_map[p2], keypoint_map[p3]))

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


app = FastAPI(title="Cow Classifier API", version="1.0.0")
extractor: CowFeatureExtractor | None = None


@app.on_event("startup")
def startup() -> None:
    global extractor
    _init_db()
    extractor = CowFeatureExtractor(MODEL_PATH)


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
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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

    return IdentifyResponse(
        recognized=recognized,
        matched_id=best_id if recognized else None,
        similarity=round(float(best_similarity), 6),
        threshold=similarity_threshold,
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
