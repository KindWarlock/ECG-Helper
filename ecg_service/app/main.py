from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .predictor import get_predictor

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="ECG Diagnosis Service", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    sampling_rate: int = Form(...),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Upload a .mat or .dat file.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mat", ".dat"}:
        raise HTTPException(status_code=400, detail="Upload a .mat or .dat file.")

    if sampling_rate <= 0:
        raise HTTPException(status_code=400, detail="Sampling rate must be positive.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            tmp_path = Path(handle.name)
            handle.write(await file.read())

        return get_predictor().predict_file(tmp_path, sampling_rate=sampling_rate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except UnboundLocalError:
            pass
