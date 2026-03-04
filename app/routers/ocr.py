import asyncio
import os
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
from fastapi import APIRouter, File, Form, UploadFile

from app.models.schemas import ImageResult, ProcessingResponse
from app.services.ocr_service import process_image, process_image_fast
from app.services.rotation import generate_rotated_images
from app.services.zip_handler import extract_images_from_zip

router = APIRouter(prefix="/api/ocr", tags=["OCR"])

_executor = ThreadPoolExecutor(max_workers=min(4, (os.cpu_count() or 2)))


def _process_single_image(img_path: str) -> ImageResult | None:
    image = cv2.imread(img_path)
    if image is None:
        return None
    filename = os.path.basename(img_path)
    ocr_result = process_image(image, filename, angle=0.0)
    return ImageResult(filename=filename, original_result=ocr_result)


@router.post("/upload", response_model=ProcessingResponse)
async def upload_zip(file: UploadFile = File(...)):
    """Upload a ZIP file and run OCR on all images inside (concurrently)."""
    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    try:
        content = await file.read()
        tmp_zip.write(content)
        tmp_zip.close()

        temp_dir, image_paths = extract_images_from_zip(tmp_zip.name)

        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(_executor, _process_single_image, img_path)
            for img_path in image_paths
        ]
        raw_results = await asyncio.gather(*tasks)
        results = [r for r in raw_results if r is not None]

        shutil.rmtree(temp_dir, ignore_errors=True)
    finally:
        os.unlink(tmp_zip.name)

    return ProcessingResponse(
        job_id=str(uuid.uuid4()),
        total_images=len(results),
        results=results,
    )


def _find_best_angle(image, filename, angle_list):
    """Run Tesseract-only OCR at all angles, return best angle and all results."""
    rotated_images = generate_rotated_images(image, angle_list)

    best_result = None
    best_score = -1.0
    all_results = []

    for angle, rotated_img in rotated_images:
        result = process_image_fast(rotated_img, filename, angle=angle)
        all_results.append((angle, rotated_img, result))

        score = result.confidence
        if result.sku:
            score += 2.0
        if result.brand:
            score += 1.0
        if result.tool_type:
            score += 0.5

        if score > best_score:
            best_score = score
            best_result = (angle, rotated_img, result)

    return best_result, all_results


def _process_with_rotation(img_path: str, angle_list: list[float]) -> ImageResult | None:
    """Two-phase rotation: fast Tesseract scan all angles, full OCR on best."""
    image = cv2.imread(img_path)
    if image is None:
        return None
    filename = os.path.basename(img_path)

    best, all_results = _find_best_angle(image, filename, angle_list)

    if best is None:
        return None

    best_angle, best_img, fast_result = best

    # If fast pass found good metadata, use it
    if fast_result.sku and fast_result.brand:
        original_result = fast_result if best_angle == 0.0 else process_image_fast(image, filename, 0.0)
        rotated_results = [r for _, _, r in all_results if r.angle != 0.0]
        return ImageResult(
            filename=filename,
            original_result=original_result,
            rotated_results=rotated_results,
        )

    # Otherwise, run full hybrid OCR only on the best angle
    full_result = process_image(best_img, filename, angle=best_angle)

    if best_angle == 0.0:
        original_result = full_result
        rotated_results = [r for _, _, r in all_results if r.angle != 0.0]
    else:
        original_result = process_image_fast(image, filename, 0.0)
        rotated_results = [
            full_result if r.angle == best_angle else r
            for _, _, r in all_results
            if r.angle != 0.0
        ]

    return ImageResult(
        filename=filename,
        original_result=original_result,
        rotated_results=rotated_results,
    )


@router.post("/upload-with-rotation", response_model=ProcessingResponse)
async def upload_zip_with_rotation(
    file: UploadFile = File(...),
    angles: str = Form(default="0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340"),
):
    """Upload a ZIP, rotate at angles, run hybrid OCR (concurrent)."""
    angle_list = [float(a.strip()) for a in angles.split(",")]

    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    try:
        content = await file.read()
        tmp_zip.write(content)
        tmp_zip.close()

        temp_dir, image_paths = extract_images_from_zip(tmp_zip.name)

        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(_executor, _process_with_rotation, img_path, angle_list)
            for img_path in image_paths
        ]
        raw_results = await asyncio.gather(*tasks)
        results = [r for r in raw_results if r is not None]

        shutil.rmtree(temp_dir, ignore_errors=True)
    finally:
        os.unlink(tmp_zip.name)

    return ProcessingResponse(
        job_id=str(uuid.uuid4()),
        total_images=len(results),
        results=results,
    )
