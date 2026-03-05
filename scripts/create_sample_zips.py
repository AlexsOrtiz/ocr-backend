#!/usr/bin/env python3
"""
Crea ZIPs de las carpetas de muestra para probar los endpoints de OCR.
Ejecutar desde la raíz del repo: python scripts/create_sample_zips.py
"""
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FOLDERS = ["CMCF801D1-JPG_R1", "Product Label", "OneDrive_1_4-3-2026"]
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
OUT_DIR = REPO_ROOT / "sample_zips"
OUT_DIR.mkdir(exist_ok=True)


def main():
    for folder_name in SAMPLE_FOLDERS:
        folder = REPO_ROOT / folder_name
        if not folder.is_dir():
            print(f"  Saltando {folder_name} (no existe)")
            continue

        zip_name = folder_name.replace(" ", "_") + ".zip"
        zip_path = OUT_DIR / zip_name
        count = 0

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() in ALLOWED_EXT and f.is_file():
                    zf.write(f, f.relative_to(REPO_ROOT))
                    count += 1

        print(f"  Creado {zip_path.name} con {count} imágenes")

    print(f"\nZIPs en: {OUT_DIR}")


if __name__ == "__main__":
    main()
