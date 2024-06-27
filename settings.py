from pathlib import Path
import sys

# Mendapatkan path absolut dari file saat ini
FILE = Path(__file__).resolve()
# Mendapatkan direktori induk dari file saat ini
ROOT = FILE.parent
# Menambahkan path root ke dalam daftar sys.path jika belum ada di sana
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Mendapatkan path relatif dari direktori root terhadap direktori kerja saat ini
ROOT = ROOT.relative_to(Path.cwd())

# Definisi sumber
IMAGE = 'Image'
WEBCAM = 'Webcam'

SOURCES_LIST = [IMAGE, WEBCAM]

# Konfigurasi gambar
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'kopi_nolabel.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'kopi_label.jpg'
UPLOADED_IMAGES_DIR = IMAGES_DIR / 'uploaded'

# Konfigurasi model ML
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'Dataset_kopi_fix.pt'

# Konfigurasi Webcam
WEBCAM_PATH = 0
