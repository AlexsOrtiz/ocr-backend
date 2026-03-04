from pydantic import BaseModel


class OCRResult(BaseModel):
    filename: str
    angle: float
    raw_text: list[str]
    sku: str | None = None
    brand: str | None = None
    model: str | None = None
    serial: str | None = None
    tool_type: str | None = None
    sbd_brand: bool = False
    confidence: float = 0.0


class ImageResult(BaseModel):
    filename: str
    original_result: OCRResult
    rotated_results: list[OCRResult] = []


class ProcessingResponse(BaseModel):
    job_id: str
    total_images: int
    results: list[ImageResult]


class RotationConfig(BaseModel):
    angles: list[float] = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
