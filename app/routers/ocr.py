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
from app.services.ocr_service import get_reader, process_image, process_image_fast, zoom_center_crop
from app.services.rotation import generate_rotated_images
from app.services.vision_ocr import extract_metadata_with_vision
from app.services.zip_handler import extract_images_from_zip

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ocr", tags=["OCR"])


ALLOWED_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif")

_executor = ThreadPoolExecutor(max_workers=2)
_fast_executor = ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4)))


@router.get("/warmup")
async def warmup():
    try:
        get_reader()
        return {"status": "ok", "message": "EasyOCR listo"}
    except Exception as e:
        logger.exception("Warmup falló: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vision-test", response_model=ProcessingResponse)
async def vision_test(file: UploadFile = File(...)):
    if not file.filename or not any(file.filename.lower().endswith(ext) for ext in ALLOWED_IMAGE_EXT):
        raise HTTPException(status_code=400, detail="Sube una imagen (.jpg, .png, .heic, etc.).")
    if not os.environ.get("OPENAI_API_KEY"):
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
            raise HTTPException(status_code=502, detail="La API de visión no devolvió resultado.")
        return ProcessingResponse(
            job_id=str(uuid.uuid4()),
            total_images=1,
            results=[ImageResult(filename=filename, original_result=ocr_result)],
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _process_single_image(img_path: str) -> ImageResult | None:
    image = load_image(img_path)
    if image is None:
        return None
    filename = os.path.basename(img_path)
    ocr_result = process_image(image, filename, angle=0.0)
    return ImageResult(filename=filename, original_result=ocr_result)


def _process_single_image_fast(img_path: str) -> ImageResult | None:
    image = load_image(img_path)
    if image is None:
        return None
    filename = os.path.basename(img_path)
    ocr_result = process_image_fast(image, filename, angle=0.0)
    return ImageResult(filename=filename, original_result=ocr_result)


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


@router.post("/upload-fast", response_model=ProcessingResponse)
async def upload_zip_fast(file: UploadFile = File(...)):
    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    try:
        content = await file.read()
        tmp_zip.write(content)
        tmp_zip.close()
        temp_dir, image_paths = extract_images_from_zip(tmp_zip.name)
        if not image_paths:
            raise HTTPException(status_code=400, detail="El ZIP no contiene imágenes válidas.")
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(_fast_executor, _process_single_image_fast, img_path)
            for img_path in image_paths
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        results = [r for r in raw_results if isinstance(r, ImageResult)]
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                logger.warning("Error en %s: %s", image_paths[i] if i < len(image_paths) else i, r)
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
            raise HTTPException(status_code=400, detail="El ZIP no contiene imágenes válidas.")
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(_fast_executor, _process_with_rotation_fast, img_path, angle_list)
            for img_path in image_paths
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        results = [r for r in raw_results if isinstance(r, ImageResult)]
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                logger.warning("Error en rotación %s: %s", image_paths[i] if i < len(image_paths) else i, r)
        shutil.rmtree(temp_dir, ignore_errors=True)
    except HTTPException:
        raise
    finally:
        try:
            os.unlink(tmp_zip.name)
        except OSError:
            pass
    return ProcessingResponse(job_id=str(uuid.uuid4()), total_images=len(results), results=results)


@router.post("/upload-single", response_model=ProcessingResponse)
async def upload_single_image(
    file: UploadFile = File(...),
    fast: bool = True,
    rotate: bool = False,
    zoom: bool = False,
):
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

        if rotate:
            angle_list = [0.0, 90.0, 180.0, 270.0]
            result = await asyncio.get_event_loop().run_in_executor(
                _fast_executor,
                _process_with_rotation_fast,
                tmp.name,
                angle_list,
            )
            if result is None:
                raise HTTPException(status_code=500, detail="No se pudo procesar la imagen.")
            result.filename = filename
            result.original_result.filename = filename
            for r in result.rotated_results:
                r.filename = filename
            return ProcessingResponse(job_id=str(uuid.uuid4()), total_images=1, results=[result])
        if fast:
            ocr_result = process_image_fast(image, filename, angle=0.0, use_zoom=zoom)
        else:
            ocr_result = process_image(image, filename, angle=0.0)
        result = ImageResult(filename=filename, original_result=ocr_result)
        return ProcessingResponse(job_id=str(uuid.uuid4()), total_images=1, results=[result])
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


RESCUE_ANGLES = [0.0, 90.0, 180.0, 270.0]


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


@router.post("/upload", response_model=ProcessingResponse)
async def upload_zip(file: UploadFile = File(...)):
    tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    try:
        content = await file.read()
        tmp_zip.write(content)
        tmp_zip.close()

        temp_dir, image_paths = extract_images_from_zip(tmp_zip.name)
        if not image_paths:
            raise HTTPException(status_code=400, detail="El ZIP no contiene imágenes válidas (.jpg, .png, .heic, etc.).")

        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(_fast_executor, _process_single_image_fast, img_path)
            for img_path in image_paths
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        results: list[ImageResult | None] = [None] * len(image_paths)
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                logger.exception("Error procesando %s: %s", image_paths[i] if i < len(image_paths) else i, r)
                continue
            if r is not None:
                results[i] = r

        rescue_indices = [
            i for i, r in enumerate(results)
            if r is not None and _result_needs_rescue(r)
        ]
        for i in rescue_indices:
            path = image_paths[i]
            try:
                rescue_result = await loop.run_in_executor(
                    _executor,
                    _process_with_rotation,
                    path,
                    RESCUE_ANGLES,
                )
            except Exception as e:
                logger.warning("Rescate por rotación falló para %s: %s", path, e)
                continue
            new_best = _pick_best_result(results[i], rescue_result)
            if new_best is rescue_result:
                logger.info("Rescate por rotación mejoró resultado para %s", os.path.basename(path))
            results[i] = new_best

        rescue_zoom_indices = [
            i for i, r in enumerate(results)
            if r is not None and _result_needs_rescue(r)
        ]
        for i in rescue_zoom_indices:
            path = image_paths[i]
            try:
                zoom_result = await loop.run_in_executor(
                    _executor,
                    _process_single_image_with_zoom,
                    path,
                )
            except Exception as e:
                logger.warning("Rescate por zoom falló para %s: %s", path, e)
                continue
            new_best = _pick_best_result(results[i], zoom_result)
            if new_best is zoom_result:
                logger.info("Rescate por zoom mejoró resultado para %s", os.path.basename(path))
            results[i] = new_best

        if os.environ.get("OPENAI_API_KEY"):
            rescue_vision_indices = [
                i for i, r in enumerate(results)
                if r is not None and _result_needs_rescue(r)
            ]
            for i in rescue_vision_indices:
                path = image_paths[i]
                try:
                    vision_result = await loop.run_in_executor(
                        _executor,
                        _process_single_image_with_vision,
                        path,
                    )
                except Exception as e:
                    logger.warning("Rescate por visión falló para %s: %s", path, e)
                    continue
                new_best = _pick_best_result(results[i], vision_result)
                if new_best is vision_result:
                    logger.info("Rescate por visión mejoró resultado para %s", os.path.basename(path))
                results[i] = new_best

        results = [r for r in results if r is not None]
        shutil.rmtree(temp_dir, ignore_errors=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error en upload OCR: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error interno en OCR: {type(e).__name__}: {str(e)}",
        )
    finally:
        try:
            os.unlink(tmp_zip.name)
        except OSError:
            pass

    return ProcessingResponse(
        job_id=str(uuid.uuid4()),
        total_images=len(results),
        results=results,
    )


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


def _process_single_image_with_zoom(img_path: str) -> ImageResult | None:
    image = load_image(img_path)
    if image is None:
        return None
    filename = os.path.basename(img_path)
    zoomed = zoom_center_crop(image, crop_ratio=0.6)
    ocr_result = process_image(zoomed, filename, angle=0.0)
    return ImageResult(filename=filename, original_result=ocr_result)


def _process_single_image_with_vision(img_path: str) -> ImageResult | None:
    image = load_image(img_path)
    if image is None:
        return None
    filename = os.path.basename(img_path)
    ocr_result = extract_metadata_with_vision(image, filename)
    if ocr_result is None:
        return None
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
        return ImageResult(
            filename=filename,
            original_result=original_result,
            rotated_results=rotated_results,
        )

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
            raise HTTPException(status_code=400, detail="El ZIP no contiene imágenes válidas.")

        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(_executor, _process_with_rotation, img_path, angle_list)
            for img_path in image_paths
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                logger.exception("Error en rotación para %s", image_paths[i] if i < len(image_paths) else i)
                continue
            if r is not None:
                results.append(r)

        shutil.rmtree(temp_dir, ignore_errors=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error en upload-with-rotation: %s", e)
        raise HTTPException(status_code=500, detail=f"Error en OCR: {type(e).__name__}: {str(e)}")
    finally:
        try:
            os.unlink(tmp_zip.name)
        except OSError:
            pass

    return ProcessingResponse(
        job_id=str(uuid.uuid4()),
        total_images=len(results),
        results=results,
    )
