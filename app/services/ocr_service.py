import re

import cv2
import easyocr
import numpy as np
from PIL import Image

from app.models.schemas import OCRResult

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

_reader: easyocr.Reader | None = None
SKU_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-+/"
SKU_PREFIXES = ("CMCF", "CMC", "DCF", "DCN", "DWE", "DW", "PBT", "GV", "D")
MAX_REGION_CANDIDATES = 2
PREFIX_REPAIRS = {
    "OWE": "DWE",
    "0WE": "DWE",
    "OW": "DW",
    "0W": "DW",
    "OCF": "DCF",
    "0CF": "DCF",
    "OCN": "DCN",
    "0CN": "DCN",
}
SKU_SHAPE_RULES = {
    "CMCF": (3, 4, True),
    "CMC": (3, 4, True),
    "DCF": (3, 4, False),
    "DCN": (3, 4, False),
    "DWE": (3, 4, False),
    "DW": (3, 4, False),
    "D": (5, 5, False),
    "PBT": (2, 3, True),
    "GV": (4, 4, True),
}
SUFFIX_CONFUSIONS = {
    "O": ("0", "O"),
    "Q": ("0", "Q"),
    "I": ("1", "I"),
    "L": ("1", "L"),
    "Z": ("2", "Z"),
    "A": ("4", "A"),
    "H": ("4", "H"),
    "T": ("7", "T"),
    "B": ("8", "B"),
    "G": ("8", "6", "G"),
    "S": ("8", "5", "S"),
}

# Brand detection with common OCR misreads
KNOWN_BRANDS = {
    "CRAFTSMAN": ["CRAFTSMAN", "CRAFTSMAK", "CRAFTSHAH", "RAFTSMAN", "[RAFTSMAN", "(RAFTSMAN", "CRAFTSMA"],
    "DEWALT": ["DEWALT", "DEWNALT", "DEWAIT", "DEWALL", "LOEWALT", "LOEWTA", "DĒWALT"],
    "RYOBI": ["RYOBI", "RYO3I", "RYOB1"],
    "MILWAUKEE": ["MILWAUKEE", "MILWAUKE", "MILWAUKEF"],
    "MAKITA": ["MAKITA"],
    "BOSCH": ["BOSCH", "B0SCH"],
    "BLACK+DECKER": ["BLACK+DECKER", "BLACK DECKER", "BLACKDECKER"],
    "RIDGID": ["RIDGID", "R1DGID"],
    "PORTER-CABLE": ["PORTER-CABLE", "PORTER CABLE", "PORTERCABLE"],
    "STANLEY": ["STANLEY", "STANLFY"],
    "KOBALT": ["KOBALT"],
    "HUSKY": ["HUSKY"],
    "HART": ["HART"],
    "SKIL": ["SKIL"],
    "WORX": ["WORX"],
}

SBD_BRANDS = {"CRAFTSMAN", "DEWALT", "STANLEY", "BLACK+DECKER", "PORTER-CABLE", "IRWIN", "LENOX"}

# SKU patterns based on real Gemini-labeled data:
# DEWALT: DW618, DW272, DWE402, DCF921, DCN662, DCN623, D26676
# RYOBI: PBT01B
# CRAFTSMAN: CMCF801
# MILWAUKEE: 2851-20
SKU_PATTERNS = [
    re.compile(r"\b(CM[A-Z]?[A-Z]?[0-9O][0-9OIl]{2,}[A-Z0-9]*)\b", re.IGNORECASE),  # CRAFTSMAN: CMCF801, CMCF8O1
    re.compile(r"\b(D[CW][A-Z]?[0-9OIl]{3,}[A-Z0-9]*)\b", re.IGNORECASE),            # DEWALT: DW618, DCF921, DCN662
    re.compile(r"\b(D[0-9]{4,}[A-Z0-9]*)\b", re.IGNORECASE),                          # DEWALT: D26676
    re.compile(r"\b(PBT[0-9OIl]{2,}[A-Z0-9]*)\b", re.IGNORECASE),                    # RYOBI: PBT01B, PBTOIB
    re.compile(r"\b(GV[0-9OIl]{3,}[A-Z0-9]*)\b", re.IGNORECASE),                     # MAKITA: GV5080z
    re.compile(r"\b(\d{4}-\d{2})\b"),                                                   # MILWAUKEE: 2851-20
    re.compile(r"\b(DWE?[0-9OIl]{3,}[A-Z0-9]*)\b", re.IGNORECASE),                   # DEWALT: DWE402
]

SERIAL_PATTERN = re.compile(r"\b(\d{6,})\b")

TOOL_TYPES = {
    "MITER SAW": "Miter Saw",
    "IMPACT DRIVER": "Impact Driver",
    "IMPACT WRENCH": "Impact Wrench",
    "COMPACT WRENCH": "Compact Wrench",
    "ROUTER": "Electronic Router",
    "PLANER": "Planer",
    "DRYWALL SCREW": "VSR Drywall Screwdriver",
    "SCREWDRIVER": "Screwdriver",
    "ANGLE GRINDER": "Angle Grinder",
    "GRINDER": "Angle Grinder",
    "FINISH NAILER": "Finish Nailer",
    "PIN NAILER": "Pin Nailer",
    "NAILER": "Nailer",
    "SANDER": "Sander",
    "DRILL": "Drill",
    "CIRCULAR SAW": "Circular Saw",
    "JIGSAW": "Jigsaw",
    "RECIPROCATING SAW": "Reciprocating Saw",
}


def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def preprocess_for_tesseract(image: np.ndarray) -> np.ndarray:
    """Preprocess image for Tesseract: grayscale, denoise, binarize."""
    resized = resize_for_ocr(image, min_long_edge=2000, max_long_edge=4000)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=12)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def read_text_tesseract(image: np.ndarray) -> tuple[list[str], float]:
    """Run Tesseract OCR with PSM 6 + PSM 11 for label text extraction."""
    if not TESSERACT_AVAILABLE:
        return [], 0.0

    binary = preprocess_for_tesseract(image)
    pil_image = Image.fromarray(binary)

    text_store: dict[str, tuple[str, float]] = {}

    for psm_mode in (6, 11):
        config = f"--oem 3 --psm {psm_mode}"
        data = pytesseract.image_to_data(
            pil_image, config=config, output_type=pytesseract.Output.DICT
        )
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            conf = float(data["conf"][i])
            if conf < 0:
                continue
            norm_conf = conf / 100.0
            key = " ".join(text.split()).upper()
            if key and (key not in text_store or norm_conf > text_store[key][1]):
                text_store[key] = (" ".join(text.split()), norm_conf)

    return summarize_text_store(text_store)


def has_sufficient_metadata(texts: list[str]) -> bool:
    """Check if OCR output has SKU or brand — enough to skip EasyOCR."""
    if not texts:
        return False
    return detect_brand(texts) is not None or detect_sku(texts) is not None


def resize_for_ocr(
    image: np.ndarray,
    min_long_edge: int = 1800,
    max_long_edge: int = 3200,
) -> np.ndarray:
    """Resize an image so small labels are easier to read without exploding memory."""
    h, w = image.shape[:2]
    long_edge = max(h, w)
    scale = 1.0

    if long_edge < min_long_edge:
        scale = min_long_edge / long_edge
    elif long_edge > max_long_edge:
        scale = max_long_edge / long_edge

    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return image


def resize_for_ocr_with_scale(
    image: np.ndarray,
    min_long_edge: int = 1800,
    max_long_edge: int = 3200,
) -> tuple[np.ndarray, float]:
    """Return the resized image plus the applied scale factor."""
    h, w = image.shape[:2]
    long_edge = max(h, w)
    scale = 1.0

    if long_edge < min_long_edge:
        scale = min_long_edge / long_edge
    elif long_edge > max_long_edge:
        scale = max_long_edge / long_edge

    if scale != 1.0:
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return resized, scale

    return image, 1.0


def enhance_image(image: np.ndarray) -> np.ndarray:
    """Enhance contrast and sharpness while preserving original coordinates."""
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (adaptive contrast enhancement) to improve text readability
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Sharpen
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    # Convert back to BGR for EasyOCR
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Enhance image for better OCR: upscale + sharpen + contrast."""
    return enhance_image(resize_for_ocr(image))


def threshold_image(image: np.ndarray) -> np.ndarray:
    """Create a high-contrast variant to help text-region detection."""
    processed = enhance_image(image)
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def read_text_candidates(
    image: np.ndarray,
    *,
    allowlist: str | None = None,
    rotation_info: list[int] | None = None,
    focused: bool = False,
) -> list[tuple[list, str, float]]:
    """Run EasyOCR with parameters tuned for small labels."""
    reader = get_reader()
    results = reader.readtext(
        image,
        allowlist=allowlist,
        rotation_info=rotation_info,
        paragraph=False,
        min_size=8 if focused else 12,
        contrast_ths=0.05,
        adjust_contrast=0.7,
        text_threshold=0.6,
        low_text=0.25,
        link_threshold=0.3,
        width_ths=1.4,
        add_margin=0.18,
        canvas_size=3200,
        mag_ratio=1.8 if focused else 1.5,
    )
    return [
        (box, " ".join(text.split()), float(conf))
        for box, text, conf in results
        if text and text.strip()
    ]


def text_signal_length(text: str) -> int:
    return len(re.sub(r"[^A-Z0-9]+", "", text.upper()))


def should_run_focus_rescue(detections: list[tuple[list, str, float]]) -> bool:
    """Use a slower zoom-in pass only when the first OCR pass is weak."""
    if not detections:
        return True

    avg_conf = sum(conf for _, _, conf in detections) / len(detections)
    signal = sum(min(12, text_signal_length(text)) for _, text, _ in detections)
    return avg_conf < 0.55 or signal < 12


def box_to_rect(box: list) -> tuple[int, int, int, int]:
    xs = [int(point[0]) for point in box]
    ys = [int(point[1]) for point in box]
    return min(xs), min(ys), max(xs), max(ys)


def scale_box_to_original(box: list, scale: float) -> list:
    if scale == 1.0:
        return box

    return [
        [int(round(point[0] / scale)), int(round(point[1] / scale))]
        for point in box
    ]


def scale_region_to_original(
    region: tuple[int, int, int, int],
    scale: float,
) -> tuple[int, int, int, int]:
    if scale == 1.0:
        return region

    x1, y1, x2, y2 = region
    return (
        int(round(x1 / scale)),
        int(round(y1 / scale)),
        int(round(x2 / scale)),
        int(round(y2 / scale)),
    )


def scale_detections_to_original(
    detections: list[tuple[list, str, float]],
    scale: float,
) -> list[tuple[list, str, float]]:
    if scale == 1.0:
        return detections

    return [
        (scale_box_to_original(box, scale), text, conf)
        for box, text, conf in detections
    ]


def should_merge_regions(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    gap: int = 28,
) -> bool:
    fx1, fy1, fx2, fy2 = first
    sx1, sy1, sx2, sy2 = second
    return not (
        fx2 + gap < sx1
        or sx2 + gap < fx1
        or fy2 + gap < sy1
        or sy2 + gap < fy1
    )


def merge_regions(regions: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not regions:
        return []

    pending = [tuple(region) for region in regions]
    merged = True

    while merged:
        merged = False
        next_regions: list[tuple[int, int, int, int]] = []

        while pending:
            current = pending.pop(0)
            index = 0

            while index < len(pending):
                other = pending[index]
                if should_merge_regions(current, other):
                    current = (
                        min(current[0], other[0]),
                        min(current[1], other[1]),
                        max(current[2], other[2]),
                        max(current[3], other[3]),
                    )
                    pending.pop(index)
                    merged = True
                    continue
                index += 1

            next_regions.append(current)

        pending = next_regions

    return sorted(
        pending,
        key=lambda region: (region[2] - region[0]) * (region[3] - region[1]),
        reverse=True,
    )


def regions_from_detections(
    detections: list[tuple[list, str, float]],
    image_shape: tuple[int, ...],
) -> list[tuple[int, int, int, int]]:
    h, w = image_shape[:2]
    image_area = h * w
    regions = []

    for box, _, conf in detections:
        x1, y1, x2, y2 = box_to_rect(box)
        width = x2 - x1
        height = y2 - y1
        area = width * height
        if conf < 0.02:
            continue
        if width < 16 or height < 10:
            continue
        if area < image_area * 0.0005 or area > image_area * 0.6:
            continue
        regions.append((x1, y1, x2, y2))

    return regions


def detect_text_regions(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect text-like regions even when recognition is poor."""
    reader = get_reader()
    resized_image, scale = resize_for_ocr_with_scale(image)
    horizontal_batches, _ = reader.detect(
        threshold_image(resized_image),
        low_text=0.2,
        text_threshold=0.45,
        link_threshold=0.2,
        mag_ratio=1.5,
        add_margin=0.15,
        slope_ths=0.2,
        ycenter_ths=0.7,
        width_ths=1.8,
    )

    if not horizontal_batches:
        return []

    regions = []
    for box in horizontal_batches[0]:
        if len(box) != 4:
            continue
        x1, x2, y1, y2 = (int(value) for value in box)
        if x2 - x1 < 16 or y2 - y1 < 10:
            continue
        regions.append(scale_region_to_original((x1, y1, x2, y2), scale))

    return merge_regions(regions)


def expand_region(
    region: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
    margin_ratio: float = 0.18,
    min_padding: int = 16,
) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = region
    pad_x = max(min_padding, int((x2 - x1) * margin_ratio))
    pad_y = max(min_padding, int((y2 - y1) * margin_ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    )


def crop_region(image: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = expand_region(region, image.shape)
    return image[y1:y2, x1:x2]


def generate_focus_crops(crop: np.ndarray) -> list[np.ndarray]:
    """Rotate tall/narrow label crops so text lines become horizontal before OCR."""
    h, w = crop.shape[:2]
    if h > w * 1.25:
        return [
            cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]
    return [crop]


def merge_text_store(
    store: dict[str, tuple[str, float]],
    detections: list[tuple[list, str, float]],
) -> None:
    for _, text, conf in detections:
        key = " ".join(text.split()).upper()
        if not key:
            continue
        if text_signal_length(key) < 2 and conf < 0.7:
            continue
        if key not in store or conf > store[key][1]:
            store[key] = (" ".join(text.split()), conf)


def summarize_text_store(store: dict[str, tuple[str, float]]) -> tuple[list[str], float]:
    ordered = sorted(
        store.values(),
        key=lambda item: (item[1], text_signal_length(item[0])),
        reverse=True,
    )
    texts = [text for text, _ in ordered]

    if not ordered:
        return [], 0.0

    top_confidences = [conf for _, conf in ordered[:5]]
    avg_conf = sum(top_confidences) / len(top_confidences)
    return texts, avg_conf


def extract_text(image: np.ndarray) -> tuple[list[str], float]:
    """Hybrid OCR: Tesseract first (fast), EasyOCR fallback if needed."""
    # --- FAST PATH: Tesseract ---
    tess_texts, tess_conf = read_text_tesseract(image)

    if has_sufficient_metadata(tess_texts):
        return tess_texts, tess_conf

    # --- SLOW PATH: EasyOCR fallback ---
    text_store: dict[str, tuple[str, float]] = {}
    # Seed with Tesseract results so they're not lost
    for text in tess_texts:
        key = text.upper()
        if key:
            text_store[key] = (text, tess_conf)

    resized_image, scale = resize_for_ocr_with_scale(image)
    primary_detections = scale_detections_to_original(
        read_text_candidates(enhance_image(resized_image)),
        scale,
    )
    merge_text_store(text_store, primary_detections)

    if should_run_focus_rescue(primary_detections):
        candidate_regions = merge_regions(
            regions_from_detections(primary_detections, image.shape) + detect_text_regions(image)
        )

        for region in candidate_regions[:MAX_REGION_CANDIDATES]:
            crop = crop_region(image, region)
            if crop.size == 0:
                continue

            for focus_crop in generate_focus_crops(crop):
                focused_variant = preprocess_image(focus_crop)
                focused_detections = read_text_candidates(
                    focused_variant,
                    focused=True,
                )
                merge_text_store(text_store, focused_detections)

                sku_detections = read_text_candidates(
                    focused_variant,
                    allowlist=SKU_ALLOWLIST,
                    focused=True,
                )
                merge_text_store(text_store, sku_detections)

    return summarize_text_store(text_store)


def fix_sku_ocr(sku: str, brand_prefix_len: int = 0) -> str:
    """Fix common OCR errors in SKU codes.

    E.g., CMCF8O1 → CMCF801, PBTOIB → PBT01B, DW6I8 → DW618

    The known prefixes are pure letters. After the prefix, any O→0, I/l→1.
    """
    # Known SKU prefixes (pure letter part)
    known_prefixes = list(SKU_PREFIXES)
    prefix = ""
    upper = sku.upper()
    for p in known_prefixes:
        if upper.startswith(p):
            prefix = sku[:len(p)]
            break

    if not prefix:
        return sku

    rest = sku[len(prefix):]

    # In the rest (numeric/alphanumeric part), fix OCR misreads
    fixed = []
    for ch in rest:
        if ch in ("O", "o"):
            fixed.append("0")
        elif ch in ("I", "l"):
            fixed.append("1")
        else:
            fixed.append(ch)

    return prefix + "".join(fixed)


def normalize_sku_shape(sku: str) -> str:
    """Trim obvious OCR tail noise while preserving common SKU formats."""
    cleaned = re.sub(r"[^A-Z0-9-]+", "", sku.upper())

    for prefix in SKU_PREFIXES:
        if not cleaned.startswith(prefix):
            continue

        rules = SKU_SHAPE_RULES.get(prefix)
        if not rules:
            return cleaned

        min_digits, max_digits, allow_suffix = rules
        remainder = cleaned[len(prefix):]
        digit_match = re.match(r"(\d+)([A-Z]*)", remainder)
        if not digit_match:
            return cleaned

        digits = digit_match.group(1)
        letters = digit_match.group(2)
        trimmed_digits = digits[:max_digits]

        if len(trimmed_digits) < min_digits:
            return cleaned

        if len(digits) > max_digits:
            return prefix + trimmed_digits

        if allow_suffix and letters:
            return prefix + trimmed_digits + letters[:1]

        return prefix + trimmed_digits

    return cleaned


def expand_ambiguous_suffix(suffix: str, limit: int = 24) -> list[str]:
    """Generate a small set of likely numeric repairs for OCR-damaged SKU suffixes."""
    candidates = [""]

    for char in suffix.upper():
        options = SUFFIX_CONFUSIONS.get(char, (char,))
        next_candidates = []

        for base in candidates:
            for option in options:
                next_candidates.append(base + option)
                if len(next_candidates) >= limit:
                    break
            if len(next_candidates) >= limit:
                break

        candidates = next_candidates or candidates

    return candidates


def generate_sku_candidates(text: str) -> list[str]:
    cleaned = re.sub(r"[^A-Z0-9-]+", "", text.upper())
    if len(cleaned) < 4:
        return []

    candidates = [cleaned]

    for bad_prefix, repaired_prefix in PREFIX_REPAIRS.items():
        if cleaned.startswith(bad_prefix):
            repaired = repaired_prefix + cleaned[len(bad_prefix):]
            if repaired not in candidates:
                candidates.append(repaired)

    for candidate_root in list(candidates):
        for prefix in SKU_PREFIXES:
            if not candidate_root.startswith(prefix):
                continue

            suffix = candidate_root[len(prefix):]
            for repaired_suffix in expand_ambiguous_suffix(suffix):
                candidate = prefix + repaired_suffix
                if candidate not in candidates:
                    candidates.append(candidate)

    return candidates


def detect_brand(texts: list[str]) -> str | None:
    full_text = " ".join(texts).upper()
    normalized_text = full_text.translate(str.maketrans({
        "0": "O",
        "1": "I",
        "3": "E",
        "4": "A",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
    }))

    for brand, variants in KNOWN_BRANDS.items():
        for variant in variants:
            variant_upper = variant.upper()
            if variant_upper in full_text or variant_upper in normalized_text:
                return brand
    return None


def detect_sku(texts: list[str]) -> str | None:
    for text in texts:
        variants = [text.strip()]
        variants.extend(generate_sku_candidates(text))

        for variant in variants:
            for pattern in SKU_PATTERNS:
                match = pattern.search(variant)
                if match:
                    sku = normalize_sku_shape(fix_sku_ocr(match.group(1)))
                    if len(sku) >= 4:
                        return sku
    return None


def detect_serial(texts: list[str], sku: str | None) -> str | None:
    serials = []
    for text in texts:
        for match in SERIAL_PATTERN.finditer(text):
            serial = match.group(1)
            if sku and serial in sku:
                continue
            serials.append(serial)
    # If we have multiple number candidates, the SKU-like one may have been
    # picked as SKU already, so return the first remaining one
    return serials[0] if serials else None


def detect_tool_type(texts: list[str]) -> str | None:
    full_text = " ".join(texts).upper()
    for keyword, tool_type in TOOL_TYPES.items():
        if keyword in full_text:
            return tool_type
    return None


def parse_metadata(texts: list[str]) -> dict:
    brand = detect_brand(texts)
    sku = detect_sku(texts)
    serial = detect_serial(texts, sku)
    tool_type = detect_tool_type(texts)
    is_sbd = brand in SBD_BRANDS if brand else False

    # If brand is DEWALT but no SKU found, check if serial looks like a D-prefixed SKU
    # e.g., OCR reads "026676" but actual SKU is "D26676"
    if brand == "DEWALT" and not sku and serial and len(serial) in (5, 6):
        # Remove leading zero if present: 026676 → D26676
        sku = "D" + serial.lstrip("0")
        serial = detect_serial(texts, sku)

    # If no brand but SKU starts with known prefix, infer brand
    if not brand and sku:
        sku_upper = sku.upper()
        if sku_upper.startswith(("DW", "DC", "D2", "D3")):
            brand = "DEWALT"
            is_sbd = True
        elif sku_upper.startswith("CM"):
            brand = "CRAFTSMAN"
            is_sbd = True
        elif sku_upper.startswith("PBT"):
            brand = "RYOBI"

    return {
        "sku": sku,
        "brand": brand,
        "model": sku,
        "serial": serial,
        "tool_type": tool_type,
        "sbd_brand": is_sbd,
    }


def process_image(image: np.ndarray, filename: str, angle: float = 0.0) -> OCRResult:
    texts, confidence = extract_text(image)
    metadata = parse_metadata(texts)

    return OCRResult(
        filename=filename,
        angle=angle,
        raw_text=texts,
        sku=metadata["sku"],
        brand=metadata["brand"],
        model=metadata["model"],
        serial=metadata["serial"],
        tool_type=metadata["tool_type"],
        sbd_brand=metadata["sbd_brand"],
        confidence=round(confidence, 4),
    )


def process_image_fast(image: np.ndarray, filename: str, angle: float = 0.0) -> OCRResult:
    """Tesseract-only processing for fast rotation scanning."""
    texts, confidence = read_text_tesseract(image)
    metadata = parse_metadata(texts)

    return OCRResult(
        filename=filename,
        angle=angle,
        raw_text=texts,
        sku=metadata["sku"],
        brand=metadata["brand"],
        model=metadata["model"],
        serial=metadata["serial"],
        tool_type=metadata["tool_type"],
        sbd_brand=metadata["sbd_brand"],
        confidence=round(confidence, 4),
    )
