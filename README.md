# ECG Research and Diagnosis Project

This repository contains ECG research notebooks, trained PyTorch model checkpoint and a FastAPI diagnosis service.

The project is organized so research notebooks can run from `notebooks/` while shared runtime assets remain at the repository root for both notebooks and `ecg_service/`.

## Project layout

```text
.
├── notebooks/          # Data download, classification, segmentation, and BPM experiments
├── models/             # Saved PyTorch checkpoints used by notebooks and the service
├── ecg_service/        # FastAPI web application for ECG diagnosis
├── requirements.txt    # Python dependencies
└── README.md
```

## Main components

- `notebooks/data_download.ipynb` downloads ECG datasets from PhysioNet.
- `notebooks/classification.ipynb` trains/evaluates a Mamba-based multi-label ECG diagnosis classifier.
- `notebooks/segmentation.ipynb` explores ECG wave segmentation and saves error examples to `outputs/error_samples/`.
- `notebooks/mamba_bpm.ipynb` experiments with BPM/R-peak estimation using ECG windows and Mamba models.
- `ecg_service/` exposes a browser UI and API for uploading ECG files and running the saved classifier.

## Data and model assets

Large runtime assets **are expected** at the repository root:

- `physionet.org/` contains downloaded datasets such as LUDB, QTDB, MIT-BIH, BUT PDB, and ECG arrhythmia data.
- `models/` contains saved `.pth` checkpoints, including the classifier checkpoint used by the service.

These paths are intentionally kept at the root because `ecg_service/` and the notebooks reference them there.

## Setup

> *git cloned* [Mamba lib](https://github.com/state-spaces/mamba) is expected at the root.

Create and activate a Python environment, then install dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs the local Mamba source tree with:

```text
-e .vendor/mamba
```

For GPU training or inference, install a PyTorch build that matches the local CUDA environment if the default package is not suitable.

## Download data

Open and run `notebooks/data_download.ipynb` to download the datasets used by the research notebooks. The notebook changes the working directory to the project root before running `wget`, so downloads are written under `physionet.org/`.

## Run notebooks

Open the notebooks from `notebooks/` in VS Code or Jupyter. The primary notebooks initialize `PROJECT_ROOT` and switch the working directory to the repository root, so root-relative paths such as `models/...` and `physionet.org/...` work from inside `notebooks/`.

## Run the ECG diagnosis service

From the repository root:

```bash
source .venv/bin/activate
uvicorn ecg_service.app.main:app --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000>.

The service accepts `.mat` and `.dat` ECG uploads, prepares the signal, runs the saved classifier, and returns diagnosis probabilities. See `ecg_service/README.md` for API details.
