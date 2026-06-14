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
    sampling_rate: int | None = Form(default=None),
    header_file: UploadFile | None = File(default=None),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Upload a .mat or .dat file.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mat", ".dat"}:
        raise HTTPException(status_code=400, detail="Upload a .mat or .dat file.")

    header_path: Path | None = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            tmp_path = Path(handle.name)
            handle.write(await file.read())

        if header_file and header_file.filename:
            header_suffix = Path(header_file.filename).suffix.lower()
            if header_suffix != ".hea":
                raise HTTPException(status_code=400, detail="Header file must be a .hea file.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=header_suffix) as handle:
                header_path = Path(handle.name)
                handle.write(await header_file.read())
            sampling_rate = read_header_sampling_rate(header_path)

        if sampling_rate is None or sampling_rate <= 0:
            raise HTTPException(status_code=400, detail="Sampling rate must be positive.")

        return get_predictor().predict_file(
            tmp_path, sampling_rate=sampling_rate, header_path=header_path
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except UnboundLocalError:
            pass
        if header_path is not None:
            header_path.unlink(missing_ok=True)


def read_header_sampling_rate(header_path: Path) -> int:
    lines = header_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise HTTPException(status_code=400, detail="Header file is empty.")
    first_line = lines[0]
    parts = first_line.split()
    if len(parts) < 3:
        raise HTTPException(status_code=400, detail="Header file does not contain a sampling rate.")
    try:
        sampling_rate = int(float(parts[2]))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Header sampling rate is invalid.") from exc
    if sampling_rate <= 0:
        raise HTTPException(status_code=400, detail="Header sampling rate must be positive.")
    return sampling_rate
