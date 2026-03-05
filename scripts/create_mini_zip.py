#!/usr/bin/env python3
"""
Crea un ZIP con solo 2-3 imágenes para pruebas rápidas (curl, etc.).
Uso: python3 scripts/create_mini_zip.py [número_imágenes]
"""
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Carpeta por defecto para el mini ZIP
DEFAULT_FOLDER = REPO_ROOT / "OneDrive_1_4-3-2026"
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
OUT_DIR = REPO_ROOT / "sample_zips"
MINI_ZIP_NAME = "mini_test.zip"


def main():
    n = 3
    if len(sys.argv) > 1:
        try:
            n = max(1, min(10, int(sys.argv[1])))
        except ValueError:
            n = 3

    folder = DEFAULT_FOLDER
    if not folder.is_dir():
        folder = REPO_ROOT / "CMCF801D1-JPG_R1"
    if not folder.is_dir():
        folder = REPO_ROOT / "Product Label"
    if not folder.is_dir():
        print("No hay carpeta con imágenes (OneDrive_1_4-3-2026, CMCF801D1-JPG_R1 o Product Label).")
        sys.exit(1)

    files = sorted(
        [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXT]
    )[:n]
    if not files:
        print(f"No hay imágenes en {folder.name}")
        sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True)
    zip_path = OUT_DIR / MINI_ZIP_NAME

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.relative_to(REPO_ROOT))

    print(f"Creado {zip_path} con {len(files)} imágenes: {[f.name for f in files]}")
    print("\nRápido (solo Tesseract, responde en segundos):")
    print(f'  curl -X POST http://127.0.0.1:8000/api/ocr/upload-fast -F "file=@{zip_path}"')
    print("\nUna sola imagen (responde en segundos):")
    print(f'  curl -X POST "http://127.0.0.1:8000/api/ocr/upload-single?fast=true" -F "file=@{files[0]}"')


if __name__ == "__main__":
    main()
