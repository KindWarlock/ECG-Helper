from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly

from .model import MambaECGClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR.resolve().parent / "models" / "classifier_aug_128.pth"
CONDITIONS_PATH = ROOT_DIR / "db/ConditionNames_SNOMED-CT.csv"
TARGET_FS = 500
TARGET_LENGTH = 5000
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DEFAULT_ADC_GAIN_PER_MV = 200.0
WFDB_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
WFDB_INTEGER_PATTERN = r"[-+]?\d+"
WFDB_GAIN_FIELD_RE = re.compile(
    rf"^(?P<gain>{WFDB_NUMBER_PATTERN})?"
    rf"(?:\((?P<baseline>[-+]?\d+)\))?"
    rf"(?:/(?P<unit>[^\s/()]+))?$"
)
WFDB_SIGNAL_LINE_RE = re.compile(
    rf"^(?P<file>\S+)\s+"
    rf"(?P<format>\S+)"
    rf"(?:\s+(?P<gain_field>\S+)"
    rf"(?:\s+(?P<adc_resolution>{WFDB_INTEGER_PATTERN})"
    rf"(?:\s+(?P<adc_zero>{WFDB_INTEGER_PATTERN})"
    rf"(?:\s+(?P<initial_value>{WFDB_INTEGER_PATTERN})"
    rf"(?:\s+(?P<checksum>{WFDB_INTEGER_PATTERN})"
    rf"(?:\s+(?P<block_size>{WFDB_INTEGER_PATTERN})"
    rf"(?:\s+(?P<description>.*))?"
    rf")?)?)?)?)?)?$"
)

# Label order produced by the notebook after filtering classes with min_total=1000.
MODEL_CLASS_CODES = [
    "270492004",
    "39732003",
    "284470004",
    "164917005",
    "55827005",
    "251146004",
    "429622005",
    "428750005",
    "55930002",
    "164934002",
    "59931005",
    "426177001",
    "426783006",
    "164889003",
    "427084000",
    "164890007",
    "427393009",
    "427172004",
    "713427006",
    "10370003",
]

# Per-class validation thresholds from section 9 of classification.ipynb.
# Order matches MODEL_CLASS_CODES and the model output order.
MODEL_CLASS_THRESHOLDS = np.asarray(
    [
        0.85,  # 1 degree atrioventricular block
        0.80,  # Axis left shift
        0.65,  # atrial premature beats
        0.35,  # abnormal Q wave
        0.50,  # left ventricular hypertrophy
        0.75,  # lower voltage QRS in all lead
        0.40,  # ST drop down
        0.65,  # ST-T Change
        0.45,  # ST change
        0.30,  # T wave Change
        0.50,  # T wave opposite
        0.85,  # Sinus Bradycardia
        0.70,  # Sinus Rhythm
        0.55,  # Atrial Fibrillation
        0.50,  # Sinus Tachycardia
        0.45,  # Atrial Flutter
        0.85,  # Sinus Irregularity
        0.50,  # premature ventricular contractions
        0.80,  # complete right bundle branch block
        0.35,  # pacemaker rythm
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class Diagnosis:
    code: str
    acronym: str
    name: str
    probability: float
    threshold: float
    detected: bool


def _read_label_table() -> list[dict[str, str]]:
    df = pd.read_csv(CONDITIONS_PATH)
    df.columns = [column.strip() for column in df.columns]
    for column in ["Acronym Name", "Full Name", "Snomed_CT"]:
        df[column] = df[column].astype(str).str.strip()
    df = df.drop_duplicates("Snomed_CT", keep="first")
    rows = []
    for code in MODEL_CLASS_CODES:
        match = df.loc[df["Snomed_CT"] == code]
        if match.empty:
            rows.append({"code": code, "acronym": "", "name": code})
            continue
        row = match.iloc[0]
        acronym = str(row["Acronym Name"]).strip()
        acronym = "" if acronym.lower() == "nan" else acronym
        rows.append(
            {"code": code, "acronym": acronym, "name": str(row["Full Name"]).strip()}
        )
    return rows


@lru_cache(maxsize=1)
def get_predictor() -> "ECGPredictor":
    return ECGPredictor()


class ECGPredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = _read_label_table()
        self.model = MambaECGClassifier(
            input_channels=12, num_classes=len(self.labels)
        ).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_file(
        self,
        signal_path: Path,
        sampling_rate: int,
        header_path: Path | None = None,
        thresholds: np.ndarray = MODEL_CLASS_THRESHOLDS,
    ) -> dict[str, Any]:
        raw_signal = load_ecg_file(signal_path)
        calibration = load_header_calibration(header_path, raw_signal.shape[1])
        model_signal = prepare_model_signal(raw_signal, sampling_rate)

        with torch.inference_mode():
            x = torch.from_numpy(model_signal.T[None, :, :]).to(
                self.device, dtype=torch.float32
            )
            logits = self.model(x)
            probabilities = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()

        diagnoses = [
            Diagnosis(
                code=label["code"],
                acronym=label["acronym"],
                name=label["name"],
                probability=float(probability),
                threshold=float(class_threshold),
                detected=bool(probability >= class_threshold),
            )
            for label, probability, class_threshold in zip(
                self.labels, probabilities, thresholds
            )
        ]
        diagnoses.sort(key=lambda item: item.probability, reverse=True)

        display_signal = downsample_for_display(raw_signal, sampling_rate)
        return {
            "sampling_rate": sampling_rate,
            "target_sampling_rate": TARGET_FS,
            "duration_seconds": float(raw_signal.shape[0] / sampling_rate),
            "samples": int(raw_signal.shape[0]),
            "leads": LEAD_NAMES,
            "signal": {
                "time": (
                    np.arange(display_signal.shape[0]) / display_signal.attrs["fs"]
                )
                .round(5)
                .tolist(),
                "values": np.asarray(display_signal).round(5).tolist(),
                "calibration": calibration,
            },
            "diagnoses": [diagnosis.__dict__ for diagnosis in diagnoses],
            "detected_diagnoses": [
                diagnosis.__dict__ for diagnosis in diagnoses if diagnosis.detected
            ],
        }

    def predict_mat(
        self,
        mat_path: Path,
        sampling_rate: int,
        thresholds: np.ndarray = MODEL_CLASS_THRESHOLDS,
    ) -> dict[str, Any]:
        return self.predict_file(
            mat_path, sampling_rate=sampling_rate, thresholds=thresholds
        )


def load_ecg_file(signal_path: Path) -> np.ndarray:
    suffix = signal_path.suffix.lower()
    if suffix == ".mat":
        return load_ecg_mat(signal_path)
    if suffix == ".dat":
        return load_ecg_dat(signal_path)
    raise ValueError("Unsupported file type. Upload a .mat or .dat file.")


def load_ecg_mat(mat_path: Path) -> np.ndarray:
    mat = loadmat(mat_path, squeeze_me=True)
    candidates: list[tuple[str, np.ndarray]] = []

    for key, value in mat.items():
        if key.startswith("__"):
            continue
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) or array.ndim != 2:
            continue
        if 12 in array.shape:
            candidates.append((key, array.astype(np.float32)))

    if not candidates:
        raise ValueError(
            "The .mat file must contain a numeric 2D array with 12 ECG channels."
        )

    _, signal = max(candidates, key=lambda item: item[1].size)
    if signal.shape[0] == 12 and signal.shape[1] != 12:
        signal = signal.T
    if signal.shape[1] != 12:
        raise ValueError(f"Expected 12 channels, got array shape {signal.shape}.")

    signal = np.nan_to_num(signal.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return signal


def load_ecg_dat(dat_path: Path) -> np.ndarray:
    """Load a raw 12-channel .dat ECG file.

    Expected layout is interleaved samples: ch1, ch2, ..., ch12, ch1, ... .
    The loader first treats data as little-endian int16, which is the common raw
    ECG/WFDB storage case. If that cannot form 12 channels, it tries float32.
    """
    for dtype in ("<i2", "<f4"):
        data = np.fromfile(dat_path, dtype=np.dtype(dtype))
        if data.size > 0 and data.size % 12 == 0:
            signal = data.reshape(-1, 12).astype(np.float32)
            return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    raise ValueError(
        "The .dat file must be a raw interleaved 12-channel signal stored as int16 or float32."
    )


def load_header_calibration(header_path: Path | None, channel_count: int) -> list[dict[str, Any]]:
    default = [
        {"gain": DEFAULT_ADC_GAIN_PER_MV, "baseline": 0.0, "unit": "mV", "source": "default"}
        for _ in range(channel_count)
    ]
    if header_path is None:
        return default

    lines = [
        line.strip()
        for line in header_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) < 2:
        return default

    calibration = []
    for line in lines[1 : channel_count + 1]:
        calibration.append(parse_wfdb_signal_line(line))

    if len(calibration) < channel_count:
        calibration.extend(default[len(calibration) :])
    return calibration


def parse_wfdb_signal_line(line: str) -> dict[str, Any]:
    match = WFDB_SIGNAL_LINE_RE.fullmatch(line.strip())
    if not match:
        return {"gain": DEFAULT_ADC_GAIN_PER_MV, "baseline": 0.0, "unit": "mV", "source": "default"}

    gain_field = match.group("gain_field") or ""
    adc_zero = _parse_float(match.group("adc_zero")) if match.group("adc_zero") is not None else 0.0
    gain, baseline, unit = parse_wfdb_gain_field(gain_field, adc_zero)
    return {"gain": gain, "baseline": baseline, "unit": unit, "source": "hea"}


def parse_wfdb_gain_field(
    value: str, adc_zero: float | None = None
) -> tuple[float, float, str]:
    adc_zero = adc_zero or 0.0
    match = WFDB_GAIN_FIELD_RE.fullmatch(value.strip())
    if not match:
        return DEFAULT_ADC_GAIN_PER_MV, adc_zero, "mV"

    gain_group = match.group("gain")
    gain = float(gain_group) if gain_group else DEFAULT_ADC_GAIN_PER_MV
    if gain == 0:
        gain = DEFAULT_ADC_GAIN_PER_MV
    baseline_group = match.group("baseline")
    baseline = float(baseline_group) if baseline_group is not None else adc_zero
    unit = match.group("unit") or "mV"
    return gain, baseline, unit


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def prepare_model_signal(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    if sampling_rate != TARGET_FS:
        gcd = np.gcd(int(TARGET_FS), int(sampling_rate))
        signal = resample_poly(
            signal, TARGET_FS // gcd, sampling_rate // gcd, axis=0
        ).astype(np.float32)

    signal = fix_length(signal, TARGET_LENGTH)
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    return ((signal - mean) / (std + 1e-6)).astype(np.float32)


def fix_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    if signal.shape[0] == target_length:
        return signal.astype(np.float32)
    if signal.shape[0] > target_length:
        return signal[:target_length].astype(np.float32)
    pad_width = target_length - signal.shape[0]
    return np.pad(signal, ((0, pad_width), (0, 0)), mode="constant").astype(np.float32)


class DisplaySignal(np.ndarray):
    attrs: dict[str, Any]


def downsample_for_display(
    signal: np.ndarray, sampling_rate: int, max_points: int = 3000
) -> DisplaySignal:
    step = max(1, int(np.ceil(signal.shape[0] / max_points)))
    display = signal[::step].astype(np.float32).view(DisplaySignal)
    display.attrs = {"fs": sampling_rate / step}
    return display
