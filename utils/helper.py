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

# ----------------- MODEL -----------------
def load_model(model_path: str = "yolov8n.pt") -> YOLO:
    """Load YOLOv8 model (nano by default for speed)."""
    return YOLO(model_path)

# ----------------- PROCESS FRAME -----------------
def process_frame(image, model, conf=0.5, tracker=None, tracking=False) -> Tuple:
    """
    Detect and annotate a single frame safely.
    Returns annotated image, class counts, detections.
    """
    h, w = image.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    if tracking and tracker:
        res = model.track(image, conf=conf, persist=True, tracker=tracker, imgsz=640)
    else:
        res = model.predict(image, conf=conf, imgsz=640)

    annotated = res[0].plot()
    detections = []

    if hasattr(res[0].boxes, 'xyxy') and len(res[0].boxes.xyxy) > 0:
        boxes = res[0].boxes.xyxy
        classes = res[0].boxes.cls

        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        if torch.is_tensor(classes):
            classes = classes.cpu().numpy()

        detections = [{"bbox": box.tolist(), "class": int(cls)} for box, cls in zip(boxes, classes)]
        counts = dict(Counter([res[0].names[int(cls)] for cls in classes]))
    else:
        counts = {}

    return annotated, counts, detections

# ----------------- IMAGE -----------------
def process_image(image_path: str, model, conf=0.5, tracker=None, tracking=False) -> Tuple[str, List[Dict]]:
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Cannot read image: {image_path}")

    annotated, _, detections = process_frame(image, model, conf, tracker, tracking)
    output_path = os.path.join(RESULTS_DIR, "annotated_image.jpg")
    cv2.imwrite(output_path, annotated)
    return output_path, detections

# ----------------- VIDEO -----------------
def process_video(video_path: str, model, conf=0.5, tracker=None, tracking=False,
                  output_name="annotated_video.mp4", max_frames=None) -> Tuple[str, List[List[Dict]]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    output_path = os.path.join(RESULTS_DIR, output_name)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    all_detections = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if max_frames and frame_count > max_frames:
            break

        annotated, _, detections = process_frame(frame, model, conf, tracker, tracking)
        out.write(annotated)
        all_detections.append(detections)

    cap.release()
    out.release()
    return output_path, all_detections

# ----------------- YOUTUBE -----------------
def process_youtube(url: str, model, conf=0.5) -> Tuple[str, List[List[Dict]]]:
    import yt_dlp
    ydl_opts = {
        "format": "best[ext=mp4]",
        "quiet": True,
        "outtmpl": os.path.join(UPLOADS_DIR, "%(id)s.%(ext)s")
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = ydl.prepare_filename(info)

    return process_video(downloaded_path, model, conf, output_name="youtube_annotated.mp4")

# ----------------- RTSP -----------------
def process_rtsp(url: str, model, conf=0.5, duration_sec=10) -> Tuple[str, List[List[Dict]]]:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise Exception(f"Cannot open RTSP stream: {url}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    total_frames = duration_sec * fps

    output_path = os.path.join(RESULTS_DIR, "rtsp_annotated.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    all_detections = []
    frame_count = 0
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        annotated, _, detections = process_frame(frame, model, conf)
        out.write(annotated)
        all_detections.append(detections)

    cap.release()
    out.release()
    return output_path, all_detections

# ----------------- WEBCAM -----------------
def process_webcam(model, conf=0.5, duration_sec=10) -> Tuple[str, List[List[Dict]]]:
    """Capture webcam feed for fixed duration."""
    return process_rtsp(0, model, conf, duration_sec=duration_sec)
