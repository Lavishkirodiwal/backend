# helper.py
from ultralytics import YOLO
import cv2
import os
import tempfile
from collections import Counter
from typing import Tuple, List, Dict

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "static", "results")
UPLOADS_DIR = os.path.join(BASE_DIR, "..", "static", "uploads")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
def load_model(model_path: str) -> YOLO:
    """Load YOLO model"""
    return YOLO(model_path)


# ---------------------------------------------------------
# COUNTING UTILITIES
# ---------------------------------------------------------
def _count_classes(res) -> Dict[str, int]:
    """Counts YOLO detections"""
    class_names = res[0].names
    counts = Counter([class_names[int(cls)] for cls in res[0].boxes.cls.tolist()])
    return dict(counts)


# ---------------------------------------------------------
# PROCESS ONE FRAME
# ---------------------------------------------------------
def process_frame(image, model, conf=0.5, tracker=None, tracking=False):
    """Detect and annotate a single frame"""

    if tracking and tracker:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)

    annotated = res[0].plot()

    detections = []
    for box, cls in zip(res[0].boxes.xyxy.tolist(), res[0].boxes.cls.tolist()):
        detections.append({
            "bbox": box,
            "class": int(cls)
        })

    counts = _count_classes(res)

    return annotated, counts, detections


# ---------------------------------------------------------
# IMAGE PROCESSING
# ---------------------------------------------------------
def process_image(image_path, model, conf=0.5, tracker=None, tracking=False):
    """Run YOLO on a single image"""

    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Cannot read image: {image_path}")

    annotated, counts, dets = process_frame(image, model, conf, tracker, tracking)

    output_path = os.path.join(RESULTS_DIR, "annotated_image.jpg")
    cv2.imwrite(output_path, annotated)

    return output_path, dets


# ---------------------------------------------------------
# VIDEO PROCESSING
# ---------------------------------------------------------
def process_video(video_path, model, conf=0.5, tracker=None, tracking=False, output_name="annotated_video.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    output_path = os.path.join(RESULTS_DIR, output_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _, detections = process_frame(frame, model, conf, tracker, tracking)
        out.write(annotated)
        all_detections.append(detections)

    cap.release()
    out.release()

    return output_path, all_detections


# ---------------------------------------------------------
# YOUTUBE PROCESSING
# ---------------------------------------------------------
def process_youtube(url, model, conf=0.5, tracker=None, tracking=False):
    import yt_dlp

    ydl_opts = {"format": "best[ext=mp4]", "quiet": True}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = ydl.prepare_filename(info)

    return process_video(downloaded_path, model, conf, tracker, tracking, "youtube_annotated.mp4")


# ---------------------------------------------------------
# RTSP PROCESSING
# ---------------------------------------------------------
def process_rtsp(url, model, conf=0.5, tracker=None, tracking=False):
    return process_video(url, model, conf, tracker, tracking, "rtsp_annotated.mp4")


# ---------------------------------------------------------
# WEBCAM PROCESSING
# ---------------------------------------------------------
def process_webcam(model, conf=0.5, tracker=None, tracking=False, duration_sec=10):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Cannot open webcam")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    output_path = os.path.join(RESULTS_DIR, "webcam_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    all_detections = []

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _, detections = process_frame(frame, model, conf, tracker, tracking)
        out.write(annotated)
        all_detections.append(detections)

    cap.release()
    out.release()

    return output_path, all_detections
