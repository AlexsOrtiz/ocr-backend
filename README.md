# OCR Backend — Extracción de metadatos de herramientas

API FastAPI que procesa imágenes (etiquetas de herramientas, códigos de producto) y extrae metadatos mediante OCR híbrido (Tesseract + EasyOCR): **SKU**, **marca**, **modelo**, **serial** y **tipo de herramienta**.

## Requisitos

- **Python 3.11+**
- **Tesseract OCR** instalado en el sistema (el código usa `pytesseract`).

### Instalar Tesseract

- **macOS:** `brew install tesseract`
- **Ubuntu/Debian:** `sudo apt install tesseract-ocr`
- **Windows:** [Instalador oficial](https://github.com/UB-Mannheim/tesseract/wiki)

## Instalación

```bash
# Clonar y entrar al repo
cd ocr-backend

# Crear entorno virtual (en macOS/Linux suele ser python3)
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Instalar dependencias (importante: hacerlo con el venv activado)
pip install -r requirements.txt
# Si pip no se encuentra, usa: python3 -m pip install -r requirements.txt
```

Opcional: copiar variables de entorno y ajustar si hace falta:

```bash
cp .env.example .env
```

## Uso

Levantar la API (con el venv activado):

```bash
uvicorn app.main:app --reload --reload-exclude 'venv/*'
```

`--reload-exclude 'venv/*'` evita que cambios en librerías del venv (p. ej. pillow_heif) reinicien el servidor.

Si `uvicorn` no se encuentra, usa el módulo de Python:

```bash
python3 -m uvicorn app.main:app --reload --reload-exclude 'venv/*'
```

- API: **http://127.0.0.1:8000**
- Documentación interactiva: **http://127.0.0.1:8000/docs**

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/api/ocr/upload` | Sube un **ZIP** con imágenes; ejecuta OCR (híbrido) sobre todas. |
| `POST` | `/api/ocr/upload-with-rotation` | Igual pero prueba varias rotaciones y elige la mejor antes del OCR. |

En ambos casos el cuerpo es `multipart/form-data` con un campo `file` (archivo ZIP).  
El segundo endpoint acepta opcionalmente `angles` (lista de ángulos en grados separados por comas).

### Probar con las carpetas de muestra

Incluidas en el repo:

- **`CMCF801D1-JPG_R1/`** — Imágenes JPG (ej. etiquetas tipo CRAFTSMAN).
- **`Product Label/`** — Imágenes PNG de etiquetas de producto.

Para generar ZIPs de prueba y, opcionalmente, probar la API:

```bash
# Crear ZIPs de las carpetas de muestra (python3 en macOS/Linux)
python3 scripts/create_sample_zips.py

# Probar upload (con la API levantada en otro terminal)
python3 scripts/test_upload.py
```

O manualmente: comprime una de las carpetas en un `.zip` y súbela en **http://127.0.0.1:8000/docs** con el endpoint `/api/ocr/upload` o `/api/ocr/upload-with-rotation`.

## Respuesta típica

Cada imagen devuelve algo como:

```json
{
  "filename": "CMCF801D1_R1-1.jpg",
  "original_result": {
    "sku": "CMCF801",
    "brand": "CRAFTSMAN",
    "model": "CMCF801",
    "serial": null,
    "tool_type": null,
    "sbd_brand": true,
    "confidence": 0.85
  },
  "rotated_results": []
}
```

## Estructura del proyecto

```
ocr-backend/
├── app/
│   ├── main.py           # FastAPI app, CORS
│   ├── routers/ocr.py     # Endpoints /api/ocr/*
│   ├── services/
│   │   ├── ocr_service.py # Lógica OCR (Tesseract + EasyOCR, SKU, marca, etc.)
│   │   ├── rotation.py   # Rotación de imágenes
│   │   └── zip_handler.py # Extracción de imágenes desde ZIP
│   └── models/schemas.py # Pydantic (OCRResult, ProcessingResponse, etc.)
├── CMCF801D1-JPG_R1/     # Muestra: JPG
├── Product Label/        # Muestra: PNG
├── scripts/
│   ├── create_sample_zips.py
│   └── test_upload.py
├── requirements.txt
├── .env.example
└── README.md
```

## Notas

- La primera ejecución puede tardar más porque EasyOCR descarga modelos.
- Para mejor rendimiento en imágenes con texto inclinado, usa `/api/ocr/upload-with-rotation`.
- Formatos admitidos en el ZIP: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`.
