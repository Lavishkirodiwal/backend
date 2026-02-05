# helper.py
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import torch
from collections import defaultdict
from ultralytics import YOLO
import numpy as np
import tempfile
import pytube
import ffmpeg

# -------------------------
# LOAD YOLO MODEL
# -------------------------
def load_model(model_path: str) -> YOLO:
    """
    Load a YOLOv8 model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = YOLO(model_path)
    return model

# -------------------------
# PROCESS IMAGE
# -------------------------
def process_image(
    image_path: str, 
    model: YOLO, 
    conf: float = 0.5
) -> List[Dict]:
    """
    Detect objects in an image and return detection info.
    """
    results = model.predict(image_path, conf=conf, verbose=False)
    detections = []

    for r in results:
        boxes = getattr(r, "boxes", [])
        if boxes is None:
            continue
        for box in boxes:
            xyxy = box.xyxy.cpu().numpy()[0].tolist()  # [x1, y1, x2, y2]
            conf_score = float(box.conf.item())
            cls = int(box.cls.item())
            label = r.names[cls]
            detections.append({
                "label": label,
                "confidence": conf_score,
                "bbox": xyxy,
                "class": cls
            })
    return detections

# -------------------------
# SAVE ANNOTATED IMAGE
# -------------------------
def save_annotated_image(
    image_path: str, 
    output_path: str, 
    detections: List[Dict], 
    class_names: Optional[Dict[int, str]] = None
) -> None:
    """
    Draw boxes and labels on image and save it.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image {image_path}")

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = det["label"]
        conf = det["confidence"]
        text = f"{label} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, text, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(output_path, img)

# -------------------------
# PROCESS VIDEO
# -------------------------
def process_video(
    video_path: str, 
    model: YOLO, 
    conf: float = 0.5, 
    output_dir: Optional[str] = None, 
    show: bool = False,
    output_name: str = "annotated_video.mp4",
    max_frames: Optional[int] = None
) -> Tuple[Path, List[List[Dict]]]:
    """
    Detect objects in a video and return annotated video path and detections.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_dir is None:
        output_dir = Path(tempfile.gettempdir())
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    all_detections = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_count >= max_frames:
            break

        results = model.predict(frame, conf=conf, verbose=False)
        frame_dets = []

        for r in results:
            boxes = getattr(r, "boxes", [])
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy()[0].tolist()
                conf_score = float(box.conf.item())
                cls = int(box.cls.item())
                label = r.names[cls]
                frame_dets.append({
                    "label": label,
                    "confidence": conf_score,
                    "bbox": xyxy,
                    "class": cls
                })
                # Draw on frame
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf_score:.2f}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        all_detections.append(frame_dets)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    return output_path, all_detections

# -------------------------
# PROCESS YOUTUBE
# -------------------------
def process_youtube(url: str, model: YOLO, conf: float = 0.5) -> Tuple[Path, List[List[Dict]]]:
    """
    Download YouTube video, detect objects, return annotated path and detections.
    """
    yt = pytube.YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    tmp_path = Path(tempfile.gettempdir()) / f"{yt.video_id}.mp4"
    stream.download(output_path=tmp_path.parent, filename=tmp_path.name)
    return process_video(str(tmp_path), model, conf)

# -------------------------
# PROCESS RTSP
# -------------------------
def process_rtsp(url: str, model: YOLO, conf: float = 0.5, duration_sec: int = 10) -> Tuple[Path, List[List[Dict]]]:
    """
    Capture RTSP stream for a certain duration and detect objects.
    """
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RTSP stream: {url}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(tempfile.gettempdir()) / f"rtsp_{uuid.uuid4().hex}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    all_detections = []
    frame_count = 0
    max_frames = int(fps * duration_sec)

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, verbose=False)
        frame_dets = []
        for r in results:
            boxes = getattr(r, "boxes", [])
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy()[0].tolist()
                conf_score = float(box.conf.item())
                cls = int(box.cls.item())
                label = r.names[cls]
                frame_dets.append({
                    "label": label,
                    "confidence": conf_score,
                    "bbox": xyxy,
                    "class": cls
                })
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf_score:.2f}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        all_detections.append(frame_dets)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    return output_path, all_detections

# -------------------------
# PROCESS WEBCAM
# -------------------------
def process_webcam(model: YOLO, conf: float = 0.5, duration_sec: int = 10) -> Tuple[Path, List[List[Dict]]]:
    """
    Capture webcam for a duration and detect objects.
    """
    return process_rtsp(0, model, conf, duration_sec)
