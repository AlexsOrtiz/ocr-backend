import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.services.image_loader import load_image
from app.services.zip_handler import extract_images_from_zip

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
MIN_FRAMES_FOR_3D = 8
FRAME_MAX_SIZE = 800
GIF_FRAME_DURATION_MS = 400
ROTATION_VIEW_STORAGE = _BASE_DIR / "storage" / "rotation_views"


def _ensure_storage():
    ROTATION_VIEW_STORAGE.mkdir(parents=True, exist_ok=True)


def _normalize_frame(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    if max(h, w) <= FRAME_MAX_SIZE:
        return bgr
    scale = FRAME_MAX_SIZE / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _bgr_to_pil_rgb(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def process_zip_for_rotation_view(zip_path: str) -> dict:
    temp_dir, image_paths = extract_images_from_zip(zip_path)

    if not image_paths:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError("El ZIP no contiene imágenes válidas (.jpg, .png, .heic, etc.).")

    if len(image_paths) < MIN_FRAMES_FOR_3D:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError(
            f"Imágenes insuficientes para generar vista 3D rotatoria. "
            f"Mínimo requerido: {MIN_FRAMES_FOR_3D}, recibidas: {len(image_paths)}."
        )

    _ensure_storage()
    job_id = str(uuid.uuid4())
    out_dir = ROTATION_VIEW_STORAGE / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    pil_frames = []

    try:
        for i, img_path in enumerate(image_paths):
            img = load_image(img_path)
            if img is None:
                continue
            normalized = _normalize_frame(img)
            frame_name = f"frame_{i:03d}.jpg"
            frame_path = out_dir / frame_name
            cv2.imwrite(str(frame_path), normalized, [cv2.IMWRITE_JPEG_QUALITY, 90])
            # Ruta relativa a "storage/" para servir vía /files/...
            frame_paths.append(f"rotation_views/{job_id}/{frame_name}")
            pil_frames.append(_bgr_to_pil_rgb(normalized))

        if len(pil_frames) < MIN_FRAMES_FOR_3D:
            shutil.rmtree(out_dir, ignore_errors=True)
            raise ValueError(
                f"Tras procesar, imágenes válidas insuficientes para 3D. "
                f"Mínimo: {MIN_FRAMES_FOR_3D}, válidas: {len(pil_frames)}."
            )

        gif_path = None
        if len(pil_frames) >= 2:
            gif_file = out_dir / "turntable.gif"
            pil_frames[0].save(
                str(gif_file),
                save_all=True,
                append_images=pil_frames[1:],
                loop=0,
                duration=GIF_FRAME_DURATION_MS,
            )
            gif_path = f"rotation_views/{job_id}/turntable.gif"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "job_id": job_id,
        "frame_count": len(frame_paths),
        "frame_paths": frame_paths,
        "gif_path": gif_path,
        "storage_dir": f"rotation_views/{job_id}",
    }
