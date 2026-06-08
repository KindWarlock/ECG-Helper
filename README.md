# ECG Project

Project structure for ECG data exploration, model training, saved artifacts, and the web service.

## Layout

- `notebooks/` — research, data download, classification, segmentation, and BPM notebooks.
- `models/` — saved PyTorch checkpoints used by notebooks and `ecg_service`.
- `physionet.org/` — downloaded PhysioNet datasets used by notebooks and `ecg_service`.
- `outputs/figures/` — generated figures and visual outputs.
- `outputs/error_samples/` — generated segmentation error samples.
- `docs/` — project documents and exported diagrams.
- `vendor/mamba/` — local Mamba source tree installed by `requirements.txt`.
- `ecg_service/` — web service application, intentionally left unchanged by this refactor.

## Notes

Runtime assets referenced by `ecg_service` remain at the workspace root: `models/` and `physionet.org/`.