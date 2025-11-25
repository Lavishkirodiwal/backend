from ultralytics import YOLO
import cv2
import os
import torch
from collections import Counter
from typing import List, Dict, Tuple

# ----------------- PATHS -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "static", "results")
UPLOADS_DIR = os.path.join(BASE_DIR, "..", "static", "uploads")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ======================================================
#  MODEL LOADING
#  Re-load model inside worker process for safety
# ======================================================

def load_model(model_path: str = "yolov8n.pt") -> YOLO:
    """Load YOLOv8 model (nano default)."""
    model = YOLO(model_path)

    # Force GPU if available
    if torch.cuda.is_available():
        model.to("cuda")

    return model


# ======================================================
# PROCESS ONE FRAME
# ======================================================

def process_frame(image, model, conf=0.5, tracker=None, tracking=False) -> Tuple:
    """Detect and annotate a single frame."""
    h, w = image.shape[:2]

    # Resize for stability
    if max(h, w) > 1280:
        scale = 1280 / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    # Run detection / tracking
    if tracking and tracker:
        res = model.track(image, conf=conf, persist=True, tracker=tracker, imgsz=640)
    else:
        res = model.predict(image, conf=conf, imgsz=640)

    # Annotated frame
    annotated = res[0].plot()

    detections = []
    boxes = res[0].boxes.xyxy
    classes = res[0].boxes.cls

    if len(boxes) > 0:
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        if torch.is_tensor(classes):
            classes = classes.cpu().numpy()

        detections = [{"bbox": box.tolist(), "class": int(cls)} for box, cls in zip(boxes, classes)]
        counts = dict(Counter([res[0].names[int(cls)] for cls in classes]))
    else:
        counts = {}

    return annotated, counts, detections


# ======================================================
# IMAGE PROCESSING
# ======================================================

def process_image(image_path: str, model, conf=0.5, tracker=None, tracking=False):
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Unable to read: {image_path}")

    annotated, _, detections = process_frame(image, model, conf, tracker, tracking)

    output_path = os.path.join(RESULTS_DIR, "annotated_image.jpg")
    cv2.imwrite(output_path, annotated)

    return output_path, detections


# ======================================================
# VIDEO PROCESSING
# ======================================================

def process_video(
    video_path: str,
    model,
    conf=0.5,
    tracker=None,
    tracking=False,
    output_name="annotated_video.mp4",
    max_frames=None
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS) or 20)

    output_path = os.path.join(RESULTS_DIR, output_name)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    all_detections = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        if max_frames and frame_index > max_frames:
            break

        annotated, _, detections = process_frame(frame, model, conf, tracker, tracking)
        out.write(annotated)
        all_detections.append(detections)

    cap.release()
    out.release()
    return output_path, all_detections


# ======================================================
# YOUTUBE
# ======================================================

def process_youtube(url: str, model, conf=0.5):
    import yt_dlp

    ydl_opts = {
        "format": "best[ext=mp4]",
        "quiet": True,
        "outtmpl": os.path.join(UPLOADS_DIR, "%(id)s.%(ext)s")
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info)

    return process_video(video_path, model, conf, output_name="youtube_annotated.mp4")


# ======================================================
# RTSP
# ======================================================

def process_rtsp(url: str, model, conf=0.5, duration_sec=10):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise Exception(f"Cannot open RTSP: {url}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS) or 20)
    total_frames = duration_sec * fps

    output_path = os.path.join(RESULTS_DIR, "rtsp_annotated.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    all_detections = []
    count = 0

    while count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1

        annotated, _, detections = process_frame(frame, model, conf)
        out.write(annotated)
        all_detections.append(detections)

    cap.release()
    out.release()
    return output_path, all_detections


# ======================================================
# WEBCAM (FIXED)
# ======================================================

def process_webcam(model, conf=0.5, duration_sec=10):
    """
    Proper webcam processing.
    DO NOT use RTSP(0).
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Cannot access webcam")

    fps = int(cap.get(cv2.CAP_PROP_FPS) or 20)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(RESULTS_DIR, "webcam_annotated.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    total_frames = duration_sec * fps

    all_detections = []
    count = 0

    while count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1

        annotated, _, detections = process_frame(frame, model, conf)
        out.write(annotated)
        all_detections.append(detections)

    cap.release()
    out.release()
    return output_path, all_detections
