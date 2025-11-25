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

from utils.settings import DETECTION_MODEL
from utils import helper as helper_api

app = FastAPI(title="YOLOv8 Object Detection API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Load YOLO model once
logger.info("Loading YOLO model...")
model = helper_api.load_model(str(DETECTION_MODEL))
CLASS_NAMES = model.names
logger.info(f"Model loaded: {DETECTION_MODEL}, classes: {CLASS_NAMES}")

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

def log_video_progress(frame_idx, total_frames, detections):
    detected_objects = sum(len(d) for d in detections)
    logger.info(f"Frame {frame_idx}/{total_frames} processed, detected objects: {detected_objects}")

# -------------------------
# IMAGE DETECTION
# -------------------------
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...), conf: float = Form(0.5)):
    start_time = time.time()
    logger.info(f"Received image detection request: {file.filename}, conf={conf}")
    try:
        temp_path = Path(tempfile.gettempdir()) / file.filename
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Saved uploaded image to {temp_path}")

        output_path, detections = await asyncio.get_event_loop().run_in_executor(
            None, helper_api.process_image, str(temp_path), model, conf
        )

        counts = count_objects(detections)
        logger.info(f"Image detection complete: {len(detections)} objects detected")
        logger.info(f"Annotated image saved to {output_path}")
        logger.info(f"Processing time: {time.time() - start_time:.2f}s")

        return JSONResponse({
            "status": "success",
            "detections": detections,
            "counts": counts,
            "annotated_image": f"/static/results/{os.path.basename(output_path)}"
        })
    except Exception as e:
        logger.exception("Error during image detection")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# -------------------------
# VIDEO DETECTION
# -------------------------
@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...), conf: float = Form(0.5), max_frames: int = Form(None)):
    start_time = time.time()
    logger.info(f"Received video detection request: {file.filename}, conf={conf}, max_frames={max_frames}")
    try:
        temp_path = Path(tempfile.gettempdir()) / file.filename
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Saved uploaded video to {temp_path}")

        output_path, all_detections = await asyncio.get_event_loop().run_in_executor(
            None, helper_api.process_video, str(temp_path), model, conf, None, False, "annotated_video.mp4", max_frames
        )

        counts = count_objects_video(all_detections)
        logger.info(f"Video detection complete: {sum(len(f) for f in all_detections)} objects detected")
        logger.info(f"Annotated video saved to {output_path}")
        logger.info(f"Total processing time: {time.time() - start_time:.2f}s")

        return JSONResponse({
            "status": "success",
            "counts": counts,
            "annotated_video": f"/static/results/{os.path.basename(output_path)}"
        })
    except Exception as e:
        logger.exception("Error during video detection")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# -------------------------
# YOUTUBE DETECTION
# -------------------------
@app.post("/detect/youtube")
async def detect_youtube(url: str = Form(...), conf: float = Form(0.5)):
    start_time = time.time()
    logger.info(f"Received YouTube detection request: {url}, conf={conf}")
    try:
        output_path, all_detections = await asyncio.get_event_loop().run_in_executor(
            None, helper_api.process_youtube, url, model, conf
        )

        counts = count_objects_video(all_detections)
        logger.info(f"YouTube detection complete: {sum(len(f) for f in all_detections)} objects detected")
        logger.info(f"Annotated video saved to {output_path}")
        logger.info(f"Total processing time: {time.time() - start_time:.2f}s")

        return JSONResponse({
            "status": "success",
            "counts": counts,
            "annotated_video": f"/static/results/{os.path.basename(output_path)}"
        })
    except Exception as e:
        logger.exception("Error during YouTube detection")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# -------------------------
# RTSP DETECTION
# -------------------------
@app.post("/detect/rtsp")
async def detect_rtsp(url: str = Form(...), conf: float = Form(0.5), duration_sec: int = Form(10)):
    start_time = time.time()
    logger.info(f"Received RTSP detection request: {url}, conf={conf}, duration={duration_sec}s")
    try:
        output_path, all_detections = await asyncio.get_event_loop().run_in_executor(
            None, helper_api.process_rtsp, url, model, conf, duration_sec
        )

        counts = count_objects_video(all_detections)
        logger.info(f"RTSP detection complete: {sum(len(f) for f in all_detections)} objects detected")
        logger.info(f"Annotated video saved to {output_path}")
        logger.info(f"Total processing time: {time.time() - start_time:.2f}s")

        return JSONResponse({
            "status": "success",
            "counts": counts,
            "annotated_video": f"/static/results/{os.path.basename(output_path)}"
        })
    except Exception as e:
        logger.exception("Error during RTSP detection")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# -------------------------
# WEBCAM DETECTION
# -------------------------
@app.post("/detect/webcam")
async def detect_webcam(conf: float = Form(0.5), duration: int = Form(10)):
    start_time = time.time()
    logger.info(f"Received webcam detection request: conf={conf}, duration={duration}s")
    try:
        output_path, all_detections = await asyncio.get_event_loop().run_in_executor(
            None, helper_api.process_webcam, model, conf, duration
        )

        counts = count_objects_video(all_detections)
        logger.info(f"Webcam detection complete: {sum(len(f) for f in all_detections)} objects detected")
        logger.info(f"Annotated video saved to {output_path}")
        logger.info(f"Total processing time: {time.time() - start_time:.2f}s")

        return JSONResponse({
            "status": "success",
            "counts": counts,
            "annotated_video": f"/static/results/{os.path.basename(output_path)}"
        })
    except Exception as e:
        logger.exception("Error during webcam detection")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
