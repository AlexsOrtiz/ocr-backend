import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers import ocr, rotation_view

_base_dir = Path(__file__).resolve().parent.parent
load_dotenv(_base_dir / ".env")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="OCR Tool Metadata Extractor", version="1.0.0")


@app.on_event("startup")
async def warmup_ocr():
    try:
        from app.services.ocr_service import PADDLE_AVAILABLE
        if PADDLE_AVAILABLE:
            from app.services.ocr_service import get_paddle_ocr
            get_paddle_ocr()
            logging.info("PaddleOCR warmup OK")
    except Exception as e:
        logging.warning("PaddleOCR warmup skip: %s", e)
    try:
        import numpy as np
        from app.services.ocr_service import get_reader
        reader = get_reader()
        dummy = np.zeros((100, 200, 3), dtype=np.uint8)
        reader.readtext(dummy)
        logging.info("EasyOCR warmup OK")
    except Exception as e:
        logging.warning("EasyOCR warmup skip: %s", e)


origins = [o.strip() for o in (os.getenv("CORS_ORIGINS") or "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ocr.router)
app.include_router(rotation_view.router)

_storage = _base_dir / "storage"
_storage.mkdir(exist_ok=True)
app.mount("/files", StaticFiles(directory=str(_storage)), name="files")


@app.get("/")
async def root():
    return {"message": "OCR API is running", "docs": "/docs"}
