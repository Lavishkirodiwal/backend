# helper.py
import os
import cv2
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
from ultralytics import YOLO
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ----------------------
# DIRECTORIES
# ----------------------
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / ".." / "static" / "results"
UPLOADS_DIR = BASE_DIR / ".." / "static" / "uploads"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------
# LOAD YOLO MODEL
# ----------------------
def load_model(model_path: str = "yolov8n.pt") -> YOLO:
    """Load YOLOv8 model and move to GPU if available."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to("cuda")
        logger.info("Model loaded to GPU")
    else:
        logger.info("Model loaded to CPU")
    return model

# ----------------------
# PROCESS IMAGE
# ----------------------
def process_image(
    image_path: str,
    model: YOLO,
    conf: float = 0.3,
) -> Tuple[str, List[Dict]]:
    """Run YOLO detection on an image and save annotated output."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image at {image_path}")

    results = model.predict(source=image, conf=conf, verbose=False)
    detections = []

    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for i in range(len(boxes)):
            box = boxes[i]
            det = {
                "class": int(box.cls[0].item()),
                "label": str(r.names[int(box.cls[0].item())]),
                "confidence": float(box.conf[0].item()),
                "bbox": box.xyxy[0].cpu().numpy().tolist(),
                "id": None,  # images don’t have IDs
            }
            detections.append(det)

    # Save annotated image
    output_path = RESULTS_DIR / f"{Path(image_path).stem}_annotated.jpg"
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f"{det['label']} {det['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(str(output_path), image)
    logger.info(f"Annotated image saved to {output_path}")

    return str(output_path), detections

# ----------------------
# PROCESS VIDEO WITH TRACKING
# ----------------------
def process_video(
    video_path: str,
    model: YOLO,
    conf: float = 0.3,
    max_frames: Optional[int] = None
) -> Tuple[Optional[str], List[List[Dict]]]:
    """Process video, detect objects with YOLO, track them, and save annotated video."""
    all_detections = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video {video_path}")
        return None, all_detections

    output_path = RESULTS_DIR / f"{Path(video_path).stem}_annotated.mp4"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    object_counts = defaultdict(int)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        results = model.track(frame, conf=conf, persist=True, verbose=False)
        frame_detections = []

        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xy = box.xyxy.cpu().numpy()
                cls = int(box.cls.item())
                conf_score = float(box.conf.item())
                obj_id = int(box.id.item()) if getattr(box, "id", None) is not None else None

                if xy.shape[0] == 0:
                    continue
                x1, y1, x2, y2 = map(int, xy[0])

                # Track object counts
                key = f"{r.names[cls]}-{obj_id}" if obj_id is not None else f"{r.names[cls]}"
                object_counts[key] += 1

                frame_detections.append({
                    "class": cls,
                    "label": r.names[cls],
                    "confidence": conf_score,
                    "bbox": [x1, y1, x2, y2],
                    "id": obj_id,
                    "width": x2-x1,
                    "height": y2-y1,
                })

                # Draw rectangle and label
                label_text = f"{r.names[cls]}"
                if obj_id is not None:
                    label_text += f"-{obj_id}"
                label_text += f" {conf_score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x1, max(0, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"W:{x2-x1} H:{y2-y1}", (x1, y2+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        all_detections.append(frame_detections)

        # Draw total counts per class
        y_offset = 20
        counts_per_class = defaultdict(int)
        for k in object_counts.keys():
            cls_name = k.split("-")[0]
            counts_per_class[cls_name] += 1
        for cls_name, count in counts_per_class.items():
            cv2.putText(frame, f"{cls_name} total: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 25

        out.write(frame)

    cap.release()
    out.release()
    logger.info(f"Annotated video saved to {output_path}")
    return str(output_path), all_detections
