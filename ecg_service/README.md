# ECG Diagnosis Service

A FastAPI web service for 12-lead ECG `.mat` and `.dat` files.

## Features

- Upload a MATLAB `.mat` file that contains a numeric 2D ECG array with 12 channels, or a raw `.dat` file.
- Select the original sampling rate.
- View all 12 ECG leads in the browser.
- Run the saved Mamba classifier checkpoint from `models/classifier_aug_128.pth` and show diagnosis probabilities.

For `.mat`, the backend accepts arrays shaped either `(samples, 12)` or `(12, samples)`. For `.dat`, the backend expects a raw interleaved 12-channel signal stored as little-endian `int16` or `float32`. The model input is resampled to 500 Hz, standardized per lead, and cropped/padded to 5000 samples.

## Install

From the repository root:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

## Launch

From the repository root:

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>.

## API

### `POST /api/predict`

Multipart form fields:

- `file`: `.mat` or `.dat` file
- `sampling_rate`: positive integer in Hz

Returns signal samples prepared for plotting, diagnosis probabilities, and detected diagnoses using a default threshold of `0.5`.
