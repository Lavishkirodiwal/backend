from pathlib import Path
import sys

# =======================
# Resolve project root
# =======================
FILE = Path(__file__).resolve()

# backend/
ROOT = FILE.parent.parent

# Add ROOT to Python path
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.append(ROOT_STR)

# =======================
# Static Directories
# =======================
STATIC_DIR = ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)

UPLOADS_DIR = STATIC_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = STATIC_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =======================
# Sources (UI dropdown)
# =======================
IMAGE = "Image"
VIDEO = "Video"
WEBCAM = "Webcam"
RTSP = "RTSP"
YOUTUBE = "YouTube"

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTSP, YOUTUBE]

# =======================
# Images
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
# Model Weights
# =======================
MODEL_DIR = ROOT / "weights"
MODEL_DIR.mkdir(exist_ok=True)

DETECTION_MODEL = MODEL_DIR / "yolov8n.pt"
SEGMENTATION_MODEL = MODEL_DIR / "yolov8n-seg.pt"

# =======================
# Webcam
# =======================
WEBCAM_PATH = 0  # Default webcam
