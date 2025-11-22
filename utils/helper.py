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
        res = model.predict(image, conf=conf , imgsz=640)

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

    # Resize image if too large to prevent timeout (Render has 30s limit)
    height, width = image.shape[:2]
    max_size = 640
    if width > max_size or height > max_size:
        scale = min(max_size / width, max_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

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


def process_video_abortable(video_path, model, conf, check_abort_callback, max_frames=None):
    """
    Processes a video frame by frame.
    - video_path: path to input video
    - model: your YOLO model
    - conf: confidence threshold
    - check_abort_callback: function returning True if processing should stop
    - max_frames: optional, limit number of frames to process
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    frame_count = 0
    all_detections = []
    output_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if max_frames and frame_count > max_frames:
            break

        # Abort if requested
        if check_abort_callback():
            print("Processing aborted by client")
            cap.release()
            return None, []

        # Run detection on frame
        results = model(frame, conf=conf)  # replace with your model's inference
        detections = [{"class": int(cls), "bbox": bbox.tolist()} for cls, bbox in zip(results.boxes.cls, results.boxes.xyxy)]
        all_detections.append(detections)

        # Optional: annotate frame
        output_frames.append(results.plot())  # or any function that draws detections

    # Save annotated video
    output_path = video_path.replace(".mp4", "_annotated.mp4")
    if output_frames:
        height, width, _ = output_frames[0].shape
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))
        for f in output_frames:
            out.write(f)
        out.release()

    cap.release()
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
