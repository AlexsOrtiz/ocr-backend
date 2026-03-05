import logging
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import RotationViewResponse
from app.services.rotation_view_service import process_zip_for_rotation_view

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/product-view", tags=["Product 360° / 3D"])


@router.post("/upload", response_model=RotationViewResponse)
async def upload_zip_for_rotation_view(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Debe subir un archivo ZIP.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    try:
        tmp.write(await file.read())
        tmp.close()
        result = process_zip_for_rotation_view(tmp.name)
        return RotationViewResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
