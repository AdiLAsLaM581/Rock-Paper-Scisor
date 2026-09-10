import os
import base64
import logging
import random
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rps-ai-backend")

# Initialize FastAPI App
app = FastAPI(
    title="Rock Paper Scissors AI - Vision API",
    description="Production backend serving in-depth ONNX deep learning hand gesture classification for Rock Paper Scissors",
    version="2.0.0"
)

# CORS Configuration
cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
allowed_origins = [orig.strip() for orig in cors_origins_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration & Model Paths
CLASSES: List[str] = ["paper", "rock", "scissors"]
MODEL_PATH = os.environ.get("MODEL_PATH", "rps_model.onnx")

potential_paths = [
    MODEL_PATH,
    os.path.join(os.path.dirname(__file__), "..", "rps_model.onnx"),
    os.path.join(os.path.dirname(__file__), "rps_model.onnx"),
    "/app/rps_model.onnx",
    "rps_model.onnx"
]

session: Optional[ort.InferenceSession] = None
input_name: Optional[str] = None
output_name: Optional[str] = None

for path in potential_paths:
    if os.path.exists(path):
        try:
            logger.info(f"Loading ONNX model from: {path}")
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            logger.info(f"Model loaded successfully! Input: {input_name}, Output: {output_name}")
            break
        except Exception as e:
            logger.error(f"Failed loading model from {path}: {e}")

if session is None:
    logger.warning("ONNX model file could not be located during startup.")


def get_winner(player: str, computer: str) -> str:
    player = player.lower()
    computer = computer.lower()
    if player == computer:
        return "Tie"
    if (player == "rock" and computer == "scissors") or \
       (player == "scissors" and computer == "paper") or \
       (player == "paper" and computer == "rock"):
        return "Player Wins!"
    return "Computer Wins!"


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def preprocess_image(
    image_bytes: bytes,
    crop_roi: bool = False,
    roi_coords: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """
    Decodes image bytes, crops ROI if specified, resizes to 150x150,
    converts BGR to RGB, normalizes pixel values to [0.0, 1.0],
    and creates [1, 150, 150, 3] float32 tensor.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image from provided payload.")

    h, w = img.shape[:2]

    # Optional ROI crop matching frontend/main.py
    if crop_roi:
        if roi_coords:
            x1 = max(0, roi_coords.get("x1", 0))
            y1 = max(0, roi_coords.get("y1", 0))
            x2 = min(w, roi_coords.get("x2", w))
            y2 = min(h, roi_coords.get("y2", h))
            if x2 > x1 and y2 > y1:
                img = img[y1:y2, x1:x2]
        else:
            # Default center 60% crop
            crop_size = int(min(w * 0.6, h * 0.65))
            cx = (w - crop_size) // 2
            cy = (h - crop_size) // 2
            img = img[cy:cy + crop_size, cx:cx + crop_size]

    # Resize to exact model input dimension (150x150)
    img = cv2.resize(img, (150, 150))

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to float32 [0.0, 1.0]
    img = img.astype(np.float32) / 255.0

    # Expand batch dimension: [1, 150, 150, 3]
    tensor = np.expand_dims(img, axis=0)
    return tensor


def run_inference(tensor: np.ndarray) -> Dict[str, Any]:
    global session, input_name, output_name
    if session is None:
        raise RuntimeError("ONNX inference session is not initialized.")

    raw_preds = session.run([output_name], {input_name: tensor})[0][0]
    probs = softmax(raw_preds)

    best_idx = int(np.argmax(probs))
    player_move = CLASSES[best_idx]
    confidence = float(probs[best_idx])

    cpu_move = random.choice(CLASSES)
    winner = get_winner(player_move, cpu_move)

    return {
        "player_move": player_move,
        "confidence": round(confidence, 4),
        "confidence_pct": round(confidence * 100, 1),
        "cpu_move": cpu_move,
        "winner": winner,
        "probabilities": {
            CLASSES[i]: round(float(probs[i]), 4)
            for i in range(len(CLASSES))
        }
    }


class PredictRequest(BaseModel):
    image: str  # Base64 string or Data URL
    crop_roi: Optional[bool] = False
    roi_coords: Optional[Dict[str, int]] = None


@app.get("/")
def root():
    return {
        "service": "Rock Paper Scissors AI Backend",
        "status": "online",
        "version": "2.0.0",
        "model_loaded": session is not None,
        "classes": CLASSES
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if session is not None else "degraded",
        "model_loaded": session is not None
    }


@app.post("/predict")
async def predict_base64(payload: PredictRequest):
    """
    Accepts base64 encoded image string (data URL or raw base64),
    runs deep learning ONNX inference, generates CPU move, and returns result.
    """
    try:
        raw_str = payload.image
        if "," in raw_str:
            raw_str = raw_str.split(",", 1)[1]

        image_bytes = base64.b64decode(raw_str)
        tensor = preprocess_image(
            image_bytes,
            crop_roi=payload.crop_roi,
            roi_coords=payload.roi_coords
        )
        return run_inference(tensor)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    """
    Accepts multipart form-data image file upload.
    """
    try:
        contents = await file.read()
        tensor = preprocess_image(contents)
        return run_inference(tensor)
    except Exception as e:
        logger.exception("Error during file prediction")
        raise HTTPException(status_code=500, detail=f"File inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
