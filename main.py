# main.py
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import tempfile
import logging
import asyncio
import os

from utils.settings import DETECTION_MODEL
from utils import helper as helper_api

app = FastAPI(title="YOLOv8 Object Detection API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLO model once
model = helper_api.load_model(str(DETECTION_MODEL))
CLASS_NAMES = model.names


# -------------------------
# Helper functions
# -------------------------
def count_objects(detections):
    counts = {}
    for det in detections:
        cls_idx = det["class"]
        cls_name = CLASS_NAMES.get(cls_idx, str(cls_idx))
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts

def count_objects_video(all_detections):
    counts = {}
    for frame_dets in all_detections:
        for det in frame_dets:
            cls_idx = det["class"]
            cls_name = CLASS_NAMES.get(cls_idx, str(cls_idx))
            counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts


# -------------------------
# Image Detection
# -------------------------
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...), conf: float = Form(0.5)):
    try:
        temp_path = Path(tempfile.gettempdir()) / file.filename
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        output_path, detections = await asyncio.get_event_loop().run_in_executor(
            None, helper_api.process_image, str(temp_path), model, conf
        )

        counts = count_objects(detections)
        return JSONResponse({
            "status": "success",
            "detections": detections,
            "counts": counts,
            "annotated_image": f"/static/results/{os.path.basename(output_path)}"
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# -------------------------
# Video Detection (fixed)
# -------------------------
@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...), conf: float = Form(0.5)):
    """
    Fixed video processing:
    - Removes client abort callback
    - Processes video fully
    - Returns annotated video and counts
    """
    try:
        temp_path = Path(tempfile.gettempdir()) / file.filename
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Run processing in executor without abort
        output_path, all_detections = await asyncio.get_event_loop().run_in_executor(
            None, helper_api.process_video, str(temp_path), model, conf
        )

        counts = count_objects_video(all_detections)
        return JSONResponse({
            "status": "success",
            "counts": counts,
            "annotated_video": f"/static/results/{os.path.basename(output_path)}"
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# -------------------------
# YouTube Detection
# -------------------------
@app.post("/detect/youtube")
async def detect_youtube(url: str = Form(...), conf: float = Form(0.5)):
    try:
        output_path, all_detections = await asyncio.get_event_loop().run_in_executor(
            None, helper_api.process_youtube, url, model, conf
        )
        counts = count_objects_video(all_detections)
        return JSONResponse({
            "status": "success",
            "counts": counts,
            "annotated_video": f"/static/results/{os.path.basename(output_path)}"
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# -------------------------
# RTSP Detection
# -------------------------
@app.post("/detect/rtsp")
async def detect_rtsp(url: str = Form(...), conf: float = Form(0.5)):
    try:
        output_path, all_detections = await asyncio.get_event_loop().run_in_executor(
            None, helper_api.process_rtsp, url, model, conf
        )
        counts = count_objects_video(all_detections)
        return JSONResponse({
            "status": "success",
            "counts": counts,
            "annotated_video": f"/static/results/{os.path.basename(output_path)}"
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# -------------------------
# Webcam Detection
# -------------------------
@app.post("/detect/webcam")
async def detect_webcam(conf: float = Form(0.5), duration: int = Form(10)):
    try:
        output_path, all_detections = await asyncio.get_event_loop().run_in_executor(
            None, helper_api.process_webcam, model, conf, None, False, duration
        )
        counts = count_objects_video(all_detections)
        return JSONResponse({
            "status": "success",
            "counts": counts,
            "annotated_video": f"/static/results/{os.path.basename(output_path)}"
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
