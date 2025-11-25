# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import tempfile
import logging
import asyncio
import os
import time
from concurrent.futures import ProcessPoolExecutor

from utils.settings import DETECTION_MODEL
from utils import helper as helper_api

app = FastAPI(title="YOLOv8 Object Detection API")

# Mount static dir
app.mount("/static", StaticFiles(directory="static"), name="static")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# GLOBAL YOLO model (loaded once)
model = helper_api.load_model(str(DETECTION_MODEL))
CLASS_NAMES = model.names

# Use MULTIPROCESSING for heavy YOLO tasks
executor = ProcessPoolExecutor(max_workers=2)


# -------------------------
# Counting helper functions
# -------------------------
def count_objects(detections):
    counts = {}
    for det in detections:
        cls_idx = det["class"]
        name = CLASS_NAMES.get(cls_idx, str(cls_idx))
        counts[name] = counts.get(name, 0) + 1
    return counts

def count_objects_video(all_detections):
    counts = {}
    for frame in all_detections:
        for det in frame:
            cls_idx = det["class"]
            name = CLASS_NAMES.get(cls_idx, str(cls_idx))
            counts[name] = counts.get(name, 0) + 1
    return counts


# -------------------------
# IMAGE DETECTION
# -------------------------
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...), conf: float = Form(0.5)):
    start_time = time.time()
    logger.info(f"[IMAGE] {file.filename}, conf={conf}")

    try:
        temp_path = Path(tempfile.gettempdir()) / file.filename
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        loop = asyncio.get_event_loop()
        output_path, detections = await loop.run_in_executor(
            executor, helper_api.process_image, str(temp_path), model, conf
        )

        return JSONResponse({
            "status": "success",
            "detections": detections,
            "counts": count_objects(detections),
            "annotated_image": f"/static/results/{os.path.basename(output_path)}"
        })

    except Exception as e:
        logger.exception("Image detection failed")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# -------------------------
# VIDEO DETECTION
# -------------------------
@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    conf: float = Form(0.5),
    max_frames: int = Form(None)
):
    start_time = time.time()
    logger.info(f"[VIDEO] {file.filename}, conf={conf}, max_frames={max_frames}")

    try:
        temp_path = Path(tempfile.gettempdir()) / file.filename
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        loop = asyncio.get_event_loop()
        output_path, all_detections = await loop.run_in_executor(
            executor,
            helper_api.process_video,
            str(temp_path), model, conf, None, False, "annotated_video.mp4", max_frames
        )

        return JSONResponse({
            "status": "success",
            "counts": count_objects_video(all_detections),
            "annotated_video": f"/static/results/{os.path.basename(output_path)}"
        })

    except Exception as e:
        logger.exception("Video detection failed")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# -------------------------
# YOUTUBE DETECTION
# -------------------------
@app.post("/detect/youtube")
async def detect_youtube(url: str = Form(...), conf: float = Form(0.5)):
    try:
        loop = asyncio.get_event_loop()
        output_path, all_detections = await loop.run_in_executor(
            executor, helper_api.process_youtube, url, model, conf
        )

        return JSONResponse({
            "status": "success",
            "counts": count_objects_video(all_detections),
            "annotated_video": f"/static/results/{os.path.basename(output_path)}"
        })

    except Exception as e:
        logger.exception("YouTube detection failed")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# -------------------------
# RTSP DETECTION
# -------------------------
@app.post("/detect/rtsp")
async def detect_rtsp(url: str = Form(...), conf: float = Form(0.5), duration_sec: int = Form(10)):
    try:
        loop = asyncio.get_event_loop()
        output_path, all_detections = await loop.run_in_executor(
            executor, helper_api.process_rtsp, url, model, conf, duration_sec
        )

        return JSONResponse({
            "status": "success",
            "counts": count_objects_video(all_detections),
            "annotated_video": f"/static/results/{os.path.basename(output_path)}"
        })

    except Exception as e:
        logger.exception("RTSP detection failed")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# -------------------------
# WEBCAM DETECTION
# -------------------------
@app.post("/detect/webcam")
async def detect_webcam(conf: float = Form(0.5), duration: int = Form(10)):
    try:
        loop = asyncio.get_event_loop()
        output_path, all_detections = await loop.run_in_executor(
            executor, helper_api.process_webcam, model, conf, duration
        )

        return JSONResponse({
            "status": "success",
            "counts": count_objects_video(all_detections),
            "annotated_video": f"/static/results/{os.path.basename(output_path)}"
        })

    except Exception as e:
        logger.exception("Webcam detection failed")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
