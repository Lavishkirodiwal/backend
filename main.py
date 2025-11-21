from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import tempfile
import logging

# Fix imports according to new structure
from utils.settings import DETECTION_MODEL
from utils import helper as helper_api

app = FastAPI(title="YOLOv8 Object Detection API")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLO model once on startup
model = helper_api.load_model(str(DETECTION_MODEL))

# Get class names from the model
CLASS_NAMES = model.names  # Example: {0: "person", 1: "car", ...}


def count_objects(detections):
    """Convert YOLO detections to readable class count dictionary"""
    counts = {}
    for det in detections:
        cls_idx = det["class"]
        cls_name = CLASS_NAMES.get(cls_idx, str(cls_idx))
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts


def count_objects_video(all_detections):
    """Aggregate counts across multiple frames"""
    counts = {}
    for frame_dets in all_detections:
        for det in frame_dets:
            cls_idx = det["class"]
            cls_name = CLASS_NAMES.get(cls_idx, str(cls_idx))
            counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts


@app.post("/detect/image")
async def detect_image(request: Request, file: UploadFile = File(...), conf: float = Form(0.5)):
    logger.info(f"Request received: {request.method} {request.url.path}")
    try:
        temp_path = Path(tempfile.gettempdir()) / file.filename

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_path, detections = helper_api.process_image(str(temp_path), model, conf=conf)

        counts = count_objects(detections)

        return JSONResponse({
            "status": "success",
            "detections": detections,
            "counts": counts,
            "annotated_image": str(output_path)
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/detect/video")
async def detect_video(request: Request, file: UploadFile = File(...), conf: float = Form(0.5)):
    logger.info(f"Request received: {request.method} {request.url.path}")
    try:
        temp_path = Path(tempfile.gettempdir()) / file.filename

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_path, all_detections = helper_api.process_video(str(temp_path), model, conf=conf)

        counts = count_objects_video(all_detections)

        return JSONResponse({
            "status": "success",
            "counts": counts,
            "annotated_video": str(output_path)
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/detect/youtube")
async def detect_youtube(request: Request, url: str = Form(...), conf: float = Form(0.5)):
    logger.info(f"Request received: {request.method} {request.url.path}")
    try:
        output_path, all_detections = helper_api.process_youtube(url, model, conf=conf)

        counts = count_objects_video(all_detections)

        return JSONResponse({
            "status": "success",
            "counts": counts,
            "annotated_video": str(output_path)
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/detect/rtsp")
async def detect_rtsp(request: Request, url: str = Form(...), conf: float = Form(0.5)):
    logger.info(f"Request received: {request.method} {request.url.path}")
    try:
        output_path, all_detections = helper_api.process_rtsp(url, model, conf=conf)

        counts = count_objects_video(all_detections)

        return JSONResponse({
            "status": "success",
            "counts": counts,
            "annotated_video": str(output_path)
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/detect/webcam")
async def detect_webcam(request: Request, conf: float = Form(0.5), duration: int = Form(10)):
    logger.info(f"Request received: {request.method} {request.url.path}")
    try:
        output_path, all_detections = helper_api.process_webcam(
            model, conf=conf, duration_sec=duration
        )

        counts = count_objects_video(all_detections)

        return JSONResponse({
            "status": "success",
            "counts": counts,
            "annotated_video": str(output_path)
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
