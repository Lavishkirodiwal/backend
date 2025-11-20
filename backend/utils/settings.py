from pathlib import Path
import sys

# =======================
# Project Root Setup
# =======================
FILE = Path(__file__).resolve()

# ROOT points to the backend folder (parent of utils)
ROOT = FILE.parent.parent  

# Add ROOT to sys.path (for imports)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# =======================
# Sources
# =======================
IMAGE = "Image"
VIDEO = "Video"
WEBCAM = "Webcam"
RTSP = "RTSP"
YOUTUBE = "YouTube"

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTSP, YOUTUBE]

# =======================
# Images Config
# =======================
IMAGES_DIR = ROOT / "static" / "uploads"  # Use uploads as images folder
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_IMAGE = IMAGES_DIR / "office_4.jpg"
DEFAULT_DETECT_IMAGE = IMAGES_DIR / "office_4_detected.jpg"

# =======================
# Videos Config
# =======================
VIDEO_DIR = ROOT / "static" / "uploads"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS_DICT = {
    "video_1": VIDEO_DIR / "video_1.mp4",
    "video_2": VIDEO_DIR / "video_2.mp4",
    "video_3": VIDEO_DIR / "video_3.mp4",
}

# =======================
# ML Model Config
# =======================
MODEL_DIR = ROOT / "weights"
MODEL_DIR.mkdir(exist_ok=True)

DETECTION_MODEL = MODEL_DIR / "yolov8n.pt"
SEGMENTATION_MODEL = MODEL_DIR / "yolov8n-seg.pt"

# =======================
# Webcam Config
# =======================
WEBCAM_PATH = 0
