import os
import tempfile
import zipfile
from pathlib import Path

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def extract_images_from_zip(zip_path: str) -> tuple[str, list[str]]:
    """Extract images from a ZIP file into a temporary directory.

    Returns (temp_dir, list_of_image_paths).
    """
    temp_dir = tempfile.mkdtemp(prefix="ocr_")
    image_paths = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for entry in zf.namelist():
            if entry.startswith("__MACOSX") or entry.startswith("."):
                continue
            ext = Path(entry).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                continue
            zf.extract(entry, temp_dir)
            image_paths.append(os.path.join(temp_dir, entry))

    return temp_dir, sorted(image_paths)
