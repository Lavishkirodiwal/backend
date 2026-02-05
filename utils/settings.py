from pathlib import Path
import sys

# =======================
# Resolve project root
# =======================
FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent  # Assume this file is in backend/ -> go up to project root

# Add ROOT to Python path so imports work
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.append(ROOT_STR)

# =======================
# Static directories
# =======================
STATIC_DIR = ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)

UPLOADS_DIR = STATIC_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = STATIC_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =======================
# Sources (UI dropdown options)
# =======================
IMAGE = "Image"
VIDEO = "Video"
WEBCAM = "Webcam"
RTSP = "RTSP"
YOUTUBE = "YouTube"

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTSP, YOUTUBE]

# =======================
# Default images
# =======================
DEFAULT_IMAGE = UPLOADS_DIR / "office_4.jpg"
DEFAULT_DETECT_IMAGE = UPLOADS_DIR / "office_4_detected.jpg"

# =======================
# Videos
# =======================
VIDEOS_DICT = {
    "video_1": UPLOADS_DIR / "video_1.mp4",
    "video_2": UPLOADS_DIR / "video_2.mp4",
    "video_3": UPLOADS_DIR / "video_3.mp4",
}

# =======================
# Model weights
# =======================
MODEL_DIR = ROOT / "weights"
MODEL_DIR.mkdir(exist_ok=True)

DETECTION_MODEL = MODEL_DIR / "yolov8n.pt"        # YOLOv8 detection model
SEGMENTATION_MODEL = MODEL_DIR / "yolov8n-seg.pt" # YOLOv8 segmentation model

# =======================
# Webcam
# =======================
WEBCAM_PATH = 0  # Default webcam index
