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
        res = model.track(image, conf=conf, persist=True, tracker=tracker, imgsz=640)
    else:
        res = model.predict(image, conf=conf, imgsz=640)

    annotated = res[0].plot()
    detections = [{"bbox": box.tolist(), "class": int(cls)} 
                  for box, cls in zip(res[0].boxes.xyxy.tolist(), res[0].boxes.cls.tolist())]
    counts = _count_classes(res)
    return annotated, counts, detections

# ---------------------------------------------------------
# IMAGE PROCESSING
# ---------------------------------------------------------
def process_image(image_path, model, conf=0.5, tracker=None, tracking=False):
    """Run YOLO on a single image"""
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Cannot read image: {image_path}")

    # Resize large images to prevent Render timeout
    max_size = 640
    h, w = image.shape[:2]
    if w > max_size or h > max_size:
        scale = min(max_size / w, max_size / h)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    annotated, counts, detections = process_frame(image, model, conf, tracker, tracking)
    output_path = os.path.join(RESULTS_DIR, f"annotated_image.jpg")
    cv2.imwrite(output_path, annotated)
    return output_path, detections

# ---------------------------------------------------------
# VIDEO PROCESSING
# ---------------------------------------------------------
def process_video(video_path, model, conf=0.5, tracker=None, tracking=False, output_name="annotated_video.mp4"):
    """Processes video efficiently, frame by frame"""
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
# YOUTUBE VIDEO PROCESSING
# ---------------------------------------------------------
def process_youtube(url, model, conf=0.5):
    import yt_dlp

    ydl_opts = {"format": "best[ext=mp4]", "quiet": True, "outtmpl": os.path.join(UPLOADS_DIR, "%(id)s.%(ext)s")}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = ydl.prepare_filename(info)

    return process_video(downloaded_path, model, conf, output_name="youtube_annotated.mp4")

# ---------------------------------------------------------
# RTSP STREAM PROCESSING
# ---------------------------------------------------------
def process_rtsp(url, model, conf=0.5):
    """Capture RTSP stream and process for fixed duration"""
    return process_video(url, model, conf, output_name="rtsp_annotated.mp4")

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

    total_frames = duration_sec * fps
    output_path = os.path.join(RESULTS_DIR, "webcam_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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
