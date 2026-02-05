# main.py
import os
import uuid
import shutil
import tempfile
import logging
import asyncio
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from utils.helper import (
    load_model,
    process_image,
    save_annotated_image,
    process_video,
    process_youtube,
    process_rtsp,
    process_webcam
)

# -------------------------
# CONFIGURATION
# -------------------------
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULTS_DIR = BASE_DIR / "static" / "results"
MODEL_PATH = BASE_DIR / "weights" / "yolov8n.pt"
CONF_DEFAULT = 0.5

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# INIT APP
# -------------------------
app = FastAPI(title="YOLOv8 Object Detection API")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# LOAD MODEL
# -------------------------
model = load_model(str(MODEL_PATH))
CLASS_NAMES = {i: name for i, name in enumerate(model.names)}

# -------------------------
# MULTIPROCESSING EXECUTOR
# -------------------------
executor = ProcessPoolExecutor(max_workers=2)

# -------------------------
# HELPERS
# -------------------------
def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    with open(destination, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    upload_file.file.close()
    return destination

def count_objects(detections: List[Dict]) -> Dict[str, int]:
    counts = {}
    for det in detections:
        cls_name = det.get("label", str(det.get("class", "")))
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts

def count_objects_video(all_detections: List[List[Dict]]) -> Dict[str, int]:
    counts = {}
    for frame in all_detections:
        for det in frame:
            cls_name = det.get("label", str(det.get("class", "")))
            counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts

# -------------------------
# IMAGE DETECTION
# -------------------------
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...), conf: float = Form(CONF_DEFAULT)):
    conf = max(0.0, min(1.0, conf))
    file_id = uuid.uuid4().hex
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    save_upload_file(file, file_path)
    output_path = RESULTS_DIR / f"{file_id}_annotated.jpg"

    try:
        loop = asyncio.get_event_loop()
        detections: List[Dict] = await loop.run_in_executor(
            executor, process_image, str(file_path), model, conf
        )
        save_annotated_image(str(file_path), str(output_path), detections, CLASS_NAMES)

        return JSONResponse({
            "status": "success",
            "detections": detections,
            "counts": count_objects(detections),
            "annotated_image": f"/static/results/{output_path.name}"
        })

    except Exception as e:
        logger.exception("Image detection failed")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# VIDEO DETECTION
# -------------------------
@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...), conf: float = Form(CONF_DEFAULT), max_frames: Optional[int] = Form(None)):
    conf = max(0.0, min(1.0, conf))
    file_id = uuid.uuid4().hex
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    save_upload_file(file, file_path)

    try:
        loop = asyncio.get_event_loop()
        output_path, all_detections = await loop.run_in_executor(
            executor, process_video, str(file_path), model, conf, None, False, f"{file_id}_annotated.mp4", max_frames
        )

        return JSONResponse({
            "status": "success",
            "counts": count_objects_video(all_detections),
            "annotated_video": f"/static/results/{output_path.name}"
        })

    except Exception as e:
        logger.exception("Video detection failed")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# YOUTUBE DETECTION
# -------------------------
@app.post("/detect/youtube")
async def detect_youtube(url: str = Form(...), conf: float = Form(CONF_DEFAULT)):
    conf = max(0.0, min(1.0, conf))

    try:
        loop = asyncio.get_event_loop()
        output_path, all_detections = await loop.run_in_executor(
            executor, process_youtube, url, model, conf
        )

        return JSONResponse({
            "status": "success",
            "counts": count_objects_video(all_detections),
            "annotated_video": f"/static/results/{output_path.name}"
        })

    except Exception as e:
        logger.exception("YouTube detection failed")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# RTSP DETECTION
# -------------------------
@app.post("/detect/rtsp")
async def detect_rtsp(url: str = Form(...), conf: float = Form(CONF_DEFAULT), duration_sec: int = Form(10)):
    conf = max(0.0, min(1.0, conf))

    try:
        loop = asyncio.get_event_loop()
        output_path, all_detections = await loop.run_in_executor(
            executor, process_rtsp, url, model, conf, duration_sec
        )

        return JSONResponse({
            "status": "success",
            "counts": count_objects_video(all_detections),
            "annotated_video": f"/static/results/{output_path.name}"
        })

    except Exception as e:
        logger.exception("RTSP detection failed")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# WEBCAM DETECTION
# -------------------------
@app.post("/detect/webcam")
async def detect_webcam(conf: float = Form(CONF_DEFAULT), duration_sec: int = Form(10)):
    conf = max(0.0, min(1.0, conf))

    try:
        loop = asyncio.get_event_loop()
        output_path, all_detections = await loop.run_in_executor(
            executor, process_webcam, model, conf, duration_sec
        )

        return JSONResponse({
            "status": "success",
            "counts": count_objects_video(all_detections),
            "annotated_video": f"/static/results/{output_path.name}"
        })

    except Exception as e:
        logger.exception("Webcam detection failed")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# PING
# -------------------------
@app.get("/ping")
async def ping():
    return {"status": "OK"}
