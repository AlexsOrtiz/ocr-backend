import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import ImageResult, OCRResult, ProcessingResponse
from app.services.image_loader import load_image
from app.services.ocr_service import (
    PADDLE_AVAILABLE,
    get_reader,
    process_image_easyocr_simple,
    process_image_fast,
    process_image_paddle,
    zoom_center_crop,
)
from app.services.rotation import generate_rotated_images
from app.services.vision_ocr import extract_metadata_with_vision
from app.services.zip_handler import extract_images_from_zip

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ocr", tags=["OCR"])

ALLOWED_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif")
RESCUE_ANGLES = [0.0, 90.0, 180.0, 270.0]

_executor = ThreadPoolExecutor(max_workers=2)
_fast_executor = ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4)))


def _vision_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _process_single_vision(img_path: str) -> ImageResult | None:
    image = load_image(img_path)
    if image is None:
        return None
    filename = os.path.basename(img_path)
    ocr_result = extract_metadata_with_vision(image, filename)
    if ocr_result is None:
        return None
    return ImageResult(filename=filename, original_result=ocr_result)


def _process_single_image(img_path: str) -> ImageResult | None:
    image = load_image(img_path)
    if image is None:
        return None
    filename = os.path.basename(img_path)
    ocr_result = process_image_easyocr_simple(image, filename, angle=0.0)
    return ImageResult(filename=filename, original_result=ocr_result)


def _process_single_image_fast(img_path: str) -> ImageResult | None:
    image = load_image(img_path)
    if image is None:
        return None
    filename = os.path.basename(img_path)
    ocr_result = process_image_fast(image, filename, angle=0.0)
    return ImageResult(filename=filename, original_result=ocr_result)


def _process_single_image_with_zoom(img_path: str) -> ImageResult | None:
    image = load_image(img_path)
    if image is None:
        return None
    filename = os.path.basename(img_path)
    zoomed = zoom_center_crop(image, crop_ratio=0.6)
    ocr_result = process_image_easyocr_simple(zoomed, filename, angle=0.0)
    return ImageResult(filename=filename, original_result=ocr_result)


def _process_with_rotation(img_path: str, angle_list: list[float]) -> ImageResult | None:
    image = load_image(img_path)
    if image is None:
        return None
    filename = os.path.basename(img_path)
    best, all_results = _find_best_angle(image, filename, angle_list)
    if best is None:
        return None
    best_angle, best_img, fast_result = best
    if fast_result.sku and fast_result.brand:
        original_result = fast_result if best_angle == 0.0 else process_image_fast(image, filename, 0.0)
        rotated_results = [r for _, _, r in all_results if r.angle != 0.0]
        return ImageResult(filename=filename, original_result=original_result, rotated_results=rotated_results)
    full_result = process_image_easyocr_simple(best_img, filename, angle=best_angle)
    if best_angle == 0.0:
        original_result = full_result
        rotated_results = [r for _, _, r in all_results if r.angle != 0.0]
    else:
        original_result = process_image_fast(image, filename, 0.0)
        rotated_results = [
            full_result if r.angle == best_angle else r
            for _, _, r in all_results if r.angle != 0.0
        ]
    return ImageResult(filename=filename, original_result=original_result, rotated_results=rotated_results)


def _process_with_rotation_fast(img_path: str, angle_list: list[float]) -> ImageResult | None:
    image = load_image(img_path)
    if image is None:
        return None
    filename = os.path.basename(img_path)
    best, all_results = _find_best_angle(image, filename, angle_list)
    if best is None:
        return None
    best_angle, _, fast_result = best
    original_result = process_image_fast(image, filename, 0.0) if best_angle != 0.0 else fast_result
    rotated_results = [r for _, _, r in all_results if r.angle != 0.0]
    return ImageResult(filename=filename, original_result=original_result, rotated_results=rotated_results)


def _find_best_angle(image, filename, angle_list):
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
        if result.sku and result.brand:
            return best_result, all_results
    return best_result, all_results


def _result_needs_rescue(image_result: ImageResult) -> bool:
    r = image_result.original_result
    return r.sku is None or r.brand is None


def _score_ocr_result(r: OCRResult) -> float:
    score = 0.0
    if r.sku:
        score += 2.0
    if r.brand:
        score += 1.0
    if r.model:
        score += 0.5
    if r.serial:
        score += 0.3
    if r.tool_type:
        score += 0.2
    score += (r.confidence or 0.0) * 0.1
    return score


def _pick_best_result(current: ImageResult | None, candidate: ImageResult | None) -> ImageResult | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    if _score_ocr_result(candidate.original_result) > _score_ocr_result(current.original_result):
        return candidate
    return current


@router.get("/warmup")
async def warmup():
    try:
        get_reader()
        return {"status": "ok", "message": "EasyOCR listo"}
    except Exception as e:
        logger.exception("Warmup failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vision-test", response_model=ProcessingResponse)
async def vision_test(file: UploadFile = File(...)):
    if not file.filename or not any(file.filename.lower().endswith(ext) for ext in ALLOWED_IMAGE_EXT):
        raise HTTPException(status_code=400, detail="Sube una imagen (.jpg, .png, .heic, etc.).")
    if not _vision_available():
        raise HTTPException(status_code=503, detail="Vision OCR no configurado: define OPENAI_API_KEY.")
    ext = os.path.splitext(file.filename or "x")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        tmp.write(await file.read())
        tmp.close()
        image = load_image(tmp.name)
        if image is None:
            raise HTTPException(status_code=400, detail="No se pudo leer la imagen.")
        filename = os.path.basename(file.filename or "image.jpg")
        ocr_result = extract_metadata_with_vision(image, filename)
        if ocr_result is None:
            raise HTTPException(status_code=502, detail="La API de vision no devolvio resultado.")
        return ProcessingResponse(
            job_id=str(uuid.uuid4()), total_images=1,
            results=[ImageResult(filename=filename, original_result=ocr_result)],
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@router.post("/upload-single", response_model=ProcessingResponse)
async def upload_single_image(file: UploadFile = File(...)):
    if not file.filename or not any(file.filename.lower().endswith(ext) for ext in ALLOWED_IMAGE_EXT):
        raise HTTPException(status_code=400, detail="Sube una imagen (.jpg, .png, .heic, etc.).")
    ext = os.path.splitext(file.filename or "x")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        tmp.write(await file.read())
        tmp.close()
        image = load_image(tmp.name)
        if image is None:
            raise HTTPException(status_code=400, detail="No se pudo leer la imagen.")
        filename = os.path.basename(file.filename or "image.jpg")
        loop = asyncio.get_event_loop()
        result = None

        if _vision_available():
            try:
                result = await loop.run_in_executor(_executor, _process_single_vision, tmp.name)
            except Exception as e:
                logger.warning("Vision OCR failed for %s: %s", filename, e)

        if result is None or _result_needs_rescue(result):
            try:
                fast_result = await loop.run_in_executor(_fast_executor, _process_single_image_fast, tmp.name)
                result = _pick_best_result(result, fast_result)
            except Exception as e:
                logger.warning("Fast OCR failed for %s: %s", filename, e)

        if result is None:
            raise HTTPException(status_code=500, detail="No se pudo procesar la imagen.")

        if _result_needs_rescue(result):
            try:
                rotated = await loop.run_in_executor(_executor, _process_with_rotation, tmp.name, RESCUE_ANGLES)
                result = _pick_best_result(result, rotated) or result
            except Exception as e:
                logger.warning("Rotation rescue failed for %s: %s", filename, e)

        if _result_needs_rescue(result):
            try:
                full_easy = await loop.run_in_executor(_executor, _process_single_image, tmp.name)
                result = _pick_best_result(result, full_easy) or result
            except Exception as e:
                logger.warning("EasyOCR rescue failed for %s: %s", filename, e)

        if _result_needs_rescue(result):
            try:
                zoomed = await loop.run_in_executor(_executor, _process_single_image_with_zoom, tmp.name)
                result = _pick_best_result(result, zoomed) or result
            except Exception as e:
                logger.warning("Zoom rescue failed for %s: %s", filename, e)

        result.filename = filename
        result.original_result.filename = filename
        for r in result.rotated_results:
            r.filename = filename
        return ProcessingResponse(job_id=str(uuid.uuid4()), total_images=1, results=[result])
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@router.post("/upload", response_model=ProcessingResponse)
async def upload_zip(file: UploadFile = File(...)):
    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    try:
        content = await file.read()
        tmp_zip.write(content)
        tmp_zip.close()
        temp_dir, image_paths = extract_images_from_zip(tmp_zip.name)
        if not image_paths:
            raise HTTPException(status_code=400, detail="El ZIP no contiene imagenes validas.")
        loop = asyncio.get_event_loop()

        results: list[ImageResult | None] = [None] * len(image_paths)

        if _vision_available():
            vision_tasks = [
                loop.run_in_executor(_executor, _process_single_vision, path)
                for path in image_paths
            ]
            vision_raw = await asyncio.gather(*vision_tasks, return_exceptions=True)
            for i, r in enumerate(vision_raw):
                if isinstance(r, Exception):
                    logger.warning("Vision failed for %s: %s", os.path.basename(image_paths[i]), r)
                    continue
                if r is not None:
                    results[i] = r

        needs_local = [i for i, r in enumerate(results) if r is None or _result_needs_rescue(r)]
        if needs_local:
            local_tasks = [
                loop.run_in_executor(_fast_executor, _process_single_image_fast, image_paths[i])
                for i in needs_local
            ]
            local_raw = await asyncio.gather(*local_tasks, return_exceptions=True)
            for j, i in enumerate(needs_local):
                r = local_raw[j]
                if isinstance(r, Exception):
                    logger.warning("Fast OCR failed for %s: %s", os.path.basename(image_paths[i]), r)
                    continue
                results[i] = _pick_best_result(results[i], r)

        rescue_indices = [i for i, r in enumerate(results) if r is not None and _result_needs_rescue(r)]
        for i in rescue_indices:
            try:
                rescue_result = await loop.run_in_executor(_executor, _process_with_rotation, image_paths[i], RESCUE_ANGLES)
            except Exception as e:
                logger.warning("Rotation rescue failed for %s: %s", os.path.basename(image_paths[i]), e)
                continue
            results[i] = _pick_best_result(results[i], rescue_result)

        zoom_indices = [i for i, r in enumerate(results) if r is not None and _result_needs_rescue(r)]
        for i in zoom_indices:
            try:
                zoom_result = await loop.run_in_executor(_executor, _process_single_image_with_zoom, image_paths[i])
            except Exception as e:
                logger.warning("Zoom rescue failed for %s: %s", os.path.basename(image_paths[i]), e)
                continue
            results[i] = _pick_best_result(results[i], zoom_result)

        results = [r for r in results if r is not None]
        shutil.rmtree(temp_dir, ignore_errors=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload OCR error: %s", e)
        raise HTTPException(status_code=500, detail=f"Error interno en OCR: {type(e).__name__}: {str(e)}")
    finally:
        try:
            os.unlink(tmp_zip.name)
        except OSError:
            pass
    return ProcessingResponse(job_id=str(uuid.uuid4()), total_images=len(results), results=results)


@router.post("/upload-fast", response_model=ProcessingResponse)
async def upload_zip_fast(file: UploadFile = File(...)):
    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    try:
        content = await file.read()
        tmp_zip.write(content)
        tmp_zip.close()
        temp_dir, image_paths = extract_images_from_zip(tmp_zip.name)
        if not image_paths:
            raise HTTPException(status_code=400, detail="El ZIP no contiene imagenes validas.")
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(_fast_executor, _process_single_image_fast, p) for p in image_paths]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        results = [r for r in raw_results if isinstance(r, ImageResult)]
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                logger.warning("Error in %s: %s", image_paths[i] if i < len(image_paths) else i, r)
        shutil.rmtree(temp_dir, ignore_errors=True)
    except HTTPException:
        raise
    finally:
        try:
            os.unlink(tmp_zip.name)
        except OSError:
            pass
    return ProcessingResponse(job_id=str(uuid.uuid4()), total_images=len(results), results=results)


@router.post("/upload-fast-rotation", response_model=ProcessingResponse)
async def upload_zip_fast_rotation(
    file: UploadFile = File(...),
    angles: str = Form(default="0,90,180,270"),
):
    angle_list = [float(a.strip()) for a in angles.split(",")]
    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    try:
        content = await file.read()
        tmp_zip.write(content)
        tmp_zip.close()
        temp_dir, image_paths = extract_images_from_zip(tmp_zip.name)
        if not image_paths:
            raise HTTPException(status_code=400, detail="El ZIP no contiene imagenes validas.")
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(_fast_executor, _process_with_rotation_fast, p, angle_list) for p in image_paths]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        results = [r for r in raw_results if isinstance(r, ImageResult)]
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                logger.warning("Rotation error %s: %s", image_paths[i] if i < len(image_paths) else i, r)
        shutil.rmtree(temp_dir, ignore_errors=True)
    except HTTPException:
        raise
    finally:
        try:
            os.unlink(tmp_zip.name)
        except OSError:
            pass
    return ProcessingResponse(job_id=str(uuid.uuid4()), total_images=len(results), results=results)


@router.post("/upload-batch", response_model=ProcessingResponse)
async def upload_batch_images(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No se recibieron archivos.")
    tmp_paths: list[tuple[str, str]] = []
    for f in files:
        if not f.filename or not any(f.filename.lower().endswith(ext) for ext in ALLOWED_IMAGE_EXT):
            continue
        ext = os.path.splitext(f.filename or "x")[1] or ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(await f.read())
        tmp.close()
        tmp_paths.append((tmp.name, os.path.basename(f.filename or "image.jpg")))
    if not tmp_paths:
        raise HTTPException(status_code=400, detail="Ningun archivo con extension valida (.jpg, .png, .heic, etc.).")
    try:
        loop = asyncio.get_event_loop()
        results: list[ImageResult] = []

        if _vision_available():
            vision_tasks = [loop.run_in_executor(_executor, _process_single_vision, p) for p, _ in tmp_paths]
            vision_raw = await asyncio.gather(*vision_tasks, return_exceptions=True)
            for i, r in enumerate(vision_raw):
                if isinstance(r, Exception):
                    logger.warning("Vision failed for %s: %s", tmp_paths[i][1], r)
                    results.append(None)
                    continue
                if r is not None:
                    r.filename = tmp_paths[i][1]
                    r.original_result.filename = tmp_paths[i][1]
                results.append(r)
        else:
            results = [None] * len(tmp_paths)

        needs_local = [i for i, r in enumerate(results) if r is None or _result_needs_rescue(r)]
        if needs_local:
            local_tasks = [
                loop.run_in_executor(_fast_executor, _process_single_image_fast, tmp_paths[i][0])
                for i in needs_local
            ]
            local_raw = await asyncio.gather(*local_tasks, return_exceptions=True)
            for j, i in enumerate(needs_local):
                r = local_raw[j]
                if isinstance(r, Exception):
                    logger.warning("Fast OCR failed for %s: %s", tmp_paths[i][1], r)
                    continue
                if r is not None:
                    r.filename = tmp_paths[i][1]
                    r.original_result.filename = tmp_paths[i][1]
                results[i] = _pick_best_result(results[i], r)

        rescue_indices = [i for i, r in enumerate(results) if r is not None and _result_needs_rescue(r)]
        for idx in rescue_indices:
            path = tmp_paths[idx][0]
            try:
                rescue_result = await loop.run_in_executor(_executor, _process_with_rotation, path, RESCUE_ANGLES)
                results[idx] = _pick_best_result(results[idx], rescue_result)
            except Exception as e:
                logger.warning("Rotation rescue failed for %s: %s", tmp_paths[idx][1], e)

        results = [r for r in results if r is not None]
    finally:
        for path, _ in tmp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass
    return ProcessingResponse(job_id=str(uuid.uuid4()), total_images=len(results), results=results)


@router.post("/upload-with-rotation", response_model=ProcessingResponse)
async def upload_zip_with_rotation(
    file: UploadFile = File(...),
    angles: str = Form(default="0,60,120,180,240,300"),
):
    angle_list = [float(a.strip()) for a in angles.split(",")]
    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    try:
        content = await file.read()
        tmp_zip.write(content)
        tmp_zip.close()
        temp_dir, image_paths = extract_images_from_zip(tmp_zip.name)
        if not image_paths:
            raise HTTPException(status_code=400, detail="El ZIP no contiene imagenes validas.")
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(_executor, _process_with_rotation, p, angle_list) for p in image_paths]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                logger.exception("Rotation error for %s", image_paths[i] if i < len(image_paths) else i)
                continue
            if r is not None:
                results.append(r)
        shutil.rmtree(temp_dir, ignore_errors=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload-with-rotation error: %s", e)
        raise HTTPException(status_code=500, detail=f"Error en OCR: {type(e).__name__}: {str(e)}")
    finally:
        try:
            os.unlink(tmp_zip.name)
        except OSError:
            pass
    return ProcessingResponse(job_id=str(uuid.uuid4()), total_images=len(results), results=results)
