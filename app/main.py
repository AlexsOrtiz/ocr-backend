from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import ocr

app = FastAPI(title="OCR Tool Metadata Extractor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ocr.router)


@app.get("/")
async def root():
    return {"message": "OCR API is running", "docs": "/docs"}
