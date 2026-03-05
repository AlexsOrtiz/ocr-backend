#!/usr/bin/env python3
"""
Prueba el endpoint rápido POST /api/ocr/upload-fast (Tesseract solo, responde en segundos).
Opcional: /api/ocr/upload-fast-rotation con pocos ángulos para fotos con inclinación.
Asegúrate de tener la API levantada: uvicorn app.main:app --reload

Ejecutar desde la raíz del repo: python scripts/test_upload.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_ZIPS_DIR = REPO_ROOT / "sample_zips"
BASE_URL = "http://127.0.0.1:8000"


def main():
    zips = list(SAMPLE_ZIPS_DIR.glob("*.zip"))
    if not zips:
        print("No hay ZIPs de muestra. Ejecuta antes: python scripts/create_sample_zips.py")
        sys.exit(1)

    try:
        import requests
    except ImportError:
        print("Instala dependencias: pip install -r requirements.txt")
        sys.exit(1)

    for zip_path in sorted(zips):
        print(f"\n--- Subiendo {zip_path.name} ---")
        try:
            with open(zip_path, "rb") as f:
                r = requests.post(
                    f"{BASE_URL}/api/ocr/upload-fast",
                    files={"file": (zip_path.name, f, "application/zip")},
                    timeout=120,
                )
        except (OSError, requests.exceptions.RequestException) as e:
            print(f"  Error de conexión (¿API levantada en {BASE_URL}?): {e}")
            continue
        if r.status_code != 200:
            print(f"  Error {r.status_code}: {r.text[:200]}")
            continue
        data = r.json()
        print(f"  job_id: {data['job_id']}")
        print(f"  imágenes procesadas: {data['total_images']}")
        for res in data["results"][:3]:
            o = res["original_result"]
            print(f"    - {res['filename']}: sku={o.get('sku')}, brand={o.get('brand')}")
        if len(data["results"]) > 3:
            print(f"    ... y {len(data['results']) - 3} más")


if __name__ == "__main__":
    main()
