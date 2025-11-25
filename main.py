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
import time
import numpy as np

from utils.settings import DETECTION_MODEL
from utils import helper as helper_api
from utils.helper import process_video_abortable

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
# Tracker for unique persons
# -------------------------
class SortTracker:
    def __init__(self):
        from sort import Sort  # pip install sort-tracker
        self.tracker = Sort()
        self.unique_ids = set()

    def update(self, detections):
        dets = []
        for det in detections:
            bbox = det['bbox']  # [x1, y1, x2, y2]
            score = det['score']
            dets.append([bbox[0], bbox[1], bbox[2], bbox[3], score])
        dets = np.array(dets)
        tracked_objects = self.tracker.update(dets)
        for obj in tracked_objects:
            track_id = int(obj[4])
            self.unique_ids.add(track_id)
        return tracked_objects

    def count(self):
        return len(self.unique_ids)

# -------------------------
# Helper functions
# -------------------------
def count_objects_frame(detections):
    counts = {}
    for det in detections:
        cls_name = CLASS_NAMES.get(det["class"], str(det["class"]))
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts

def count_objects_video(all_detections):
    counts = {}
    for frame_dets in all_detections:
        for det in frame_dets:
            cls_name = CLASS_NAMES.get(det["class"], str(det["class"]))
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

        counts = count_objects_frame(detections)
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

        def process_video_with_logging(*args, **kwargs):
            output_path, all_dets = helper_api.process_video(*args, **kwargs)
            total_frames = len(all_dets)
            logger.info(f"Video total frames: {total_frames}")
            for idx, frame_dets in enumerate(all_dets, start=1):
                log_video_progress(idx, total_frames, [frame_dets])
            return output_path, all_dets

        output_path, all_detections = await asyncio.get_event_loop().run_in_executor(
            None, process_video_with_logging, str(temp_path), model, conf, None, False, "annotated_video.mp4", max_frames
        )

        # Unique person tracking
        tracker = SortTracker()
        for frame_dets in all_detections:
            persons = [det for det in frame_dets if det["class"] == 0]
            tracker.update(persons)
        counts = count_objects_video(all_detections)
        counts["person"] = tracker.count()

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
        def process_with_logging(url, model, conf):
            output_path, all_dets = helper_api.process_youtube(url, model, conf)
            total_frames = len(all_dets)
            logger.info(f"YouTube video total frames: {total_frames}")
            for idx, frame_dets in enumerate(all_dets, start=1):
                log_video_progress(idx, total_frames, [frame_dets])
            return output_path, all_dets

        output_path, all_detections = await asyncio.get_event_loop().run_in_executor(
            None, process_with_logging, url, model, conf
        )

        tracker = SortTracker()
        for frame_dets in all_detections:
            persons = [det for det in frame_dets if det["class"] == 0]
            tracker.update(persons)
        counts = count_objects_video(all_detections)
        counts["person"] = tracker.count()

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
        def process_with_logging(url, model, conf, duration_sec):
            output_path, all_dets = helper_api.process_rtsp(url, model, conf, duration_sec)
            total_frames = len(all_dets)
            logger.info(f"RTSP stream total frames: {total_frames}")
            for idx, frame_dets in enumerate(all_dets, start=1):
                log_video_progress(idx, total_frames, [frame_dets])
            return output_path, all_dets

        output_path, all_detections = await asyncio.get_event_loop().run_in_executor(
            None, process_with_logging, url, model, conf, duration_sec
        )

        tracker = SortTracker()
        for frame_dets in all_detections:
            persons = [det for det in frame_dets if det["class"] == 0]
            tracker.update(persons)
        counts = count_objects_video(all_detections)
        counts["person"] = tracker.count()

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
        def process_with_logging(model, conf, duration):
            output_path, all_dets = helper_api.process_webcam(model, conf, duration_sec=duration)
            total_frames = len(all_dets)
            logger.info(f"Webcam total frames: {total_frames}")
            for idx, frame_dets in enumerate(all_dets, start=1):
                log_video_progress(idx, total_frames, [frame_dets])
            return output_path, all_dets

        output_path, all_detections = await asyncio.get_event_loop().run_in_executor(
            None, process_with_logging, model, conf, duration
        )

        tracker = SortTracker()
        for frame_dets in all_detections:
            persons = [det for det in frame_dets if det["class"] == 0]
            tracker.update(persons)
        counts = count_objects_video(all_detections)
        counts["person"] = tracker.count()

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
