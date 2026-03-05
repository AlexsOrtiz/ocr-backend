import cv2
import numpy as np

_HEIC_REGISTERED = False


def _load_heic(path: str) -> np.ndarray | None:
    global _HEIC_REGISTERED
    try:
        import pillow_heif
        from PIL import Image
    except ImportError:
        return None
    if not _HEIC_REGISTERED:
        pillow_heif.register_heif_opener()
        _HEIC_REGISTERED = True
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        arr = np.array(img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def load_image(path: str) -> np.ndarray | None:
    path_lower = path.lower()
    if path_lower.endswith(".heic") or path_lower.endswith(".heif"):
        return _load_heic(path)
    return cv2.imread(path)
