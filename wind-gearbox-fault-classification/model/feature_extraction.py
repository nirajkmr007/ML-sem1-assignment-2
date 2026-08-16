"""
feature_extraction.py
---------------------
Turns raw wind-turbine gearbox accelerometer recordings into a tabular,
model-ready condition-monitoring (CM) feature set.

Why this step exists
====================
The public gearbox dataset ships as raw vibration time series: four
accelerometer channels sampled continuously while the gearbox runs at a
30 Hz input shaft speed under ten different load settings (0 %..90 %).
Raw samples are useless to a classical classifier on their own -- a single
acceleration value carries no information about gear health. Real condition
monitoring systems therefore cut the stream into short segments and reduce
each segment to a vector of *health indicators*. This module implements that
step, using descriptors that are standard practice in rotating-machinery
diagnostics (ISO 13373 style time-domain indicators plus two spectral shape
descriptors).

Output
======
A tidy DataFrame with 36 predictor columns and one target column:

    9 descriptors x 4 accelerometer channels  = 36 vibration features
    + gearbox_state                            = target (6 classes)

Choice of target
================
The raw dataset only distinguishes two health states, and that binary problem
turns out to be trivially separable here: the healthy and broken-tooth
recordings were captured in separate acquisition campaigns whose overall
vibration levels barely overlap, so *every* classifier scores ~100 %
(verified in the notebook). A table of six identical 1.000 rows says nothing
about the models, so this project targets the operationally harder question a
condition monitoring system actually has to answer:

    "From the vibration signature alone, what is the gearbox doing --
     which health state AND which load regime?"

Load-dependent behaviour matters because CM alarm thresholds are load-scheduled;
a box that can recover the operating regime from vibration alone stays useful
when the SCADA load channel is missing or out of sync. The target therefore
combines health state with a three-level load regime:

    Healthy-Low  Healthy-Mid  Healthy-High
    Broken-Low   Broken-Mid   Broken-High

``load_pct`` is deliberately NOT a predictor -- it would leak the load regime
straight into the label.

Author: Niraj Kumar -- BITS WILP M.Tech (AI/ML), Machine Learning Assignment 2
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEGMENT_LENGTH = 512          # samples per analysis window (non-overlapping)
CHANNEL_NAMES = ("a1", "a2", "a3", "a4")
TARGET_COLUMN = "gearbox_state"
HOLDOUT_FRACTION = 0.30       # last 30 % of every recording is held out

# Load regimes used to build the composite target
LOAD_REGIMES = ((0, 30, "Low"), (40, 60, "Mid"), (70, 90, "High"))

# Descriptor names, in the order produced by `channel_descriptors`
DESCRIPTORS = (
    "rms",              # overall vibration energy  -> wear / looseness
    "p2p",              # peak-to-peak swing        -> impact severity
    "kurtosis",         # 4th moment                -> impulsiveness (tooth strike)
    "skewness",         # 3rd moment                -> waveform asymmetry
    "crest_factor",     # peak / rms                -> early localised damage
    "shape_factor",     # rms / mean|x|             -> waveform shape drift
    "impulse_factor",   # peak / mean|x|            -> impulsive content
    "spec_centroid",    # spectral centre of mass   -> energy shifting up in freq
    "hf_ratio",         # >Nyquist/4 energy share   -> high-frequency excitation
)

_EPS = 1e-12


# --------------------------------------------------------------------------
# Segment-level descriptors
# --------------------------------------------------------------------------

def channel_descriptors(segment: np.ndarray) -> np.ndarray:
    """Reduce one accelerometer segment to the 9 health indicators above.

    Parameters
    ----------
    segment : 1-D array of raw acceleration samples.

    Returns
    -------
    np.ndarray of shape (9,), ordered exactly as ``DESCRIPTORS``.
    """
    x = np.asarray(segment, dtype=np.float64)
    centred = x - x.mean()

    abs_x = np.abs(x)
    rms = np.sqrt(np.mean(x ** 2))
    mean_abs = abs_x.mean()
    peak = abs_x.max()

    # --- time-domain statistical moments -------------------------------
    sigma = centred.std()
    kurtosis = np.mean(centred ** 4) / (sigma ** 4 + _EPS)
    skewness = np.mean(centred ** 3) / (sigma ** 3 + _EPS)

    # --- dimensionless diagnostic ratios -------------------------------
    crest_factor = peak / (rms + _EPS)
    shape_factor = rms / (mean_abs + _EPS)
    impulse_factor = peak / (mean_abs + _EPS)

    # --- spectral shape (single-sided magnitude spectrum) --------------
    # Frequencies are expressed as a fraction of Nyquist so that the
    # features stay valid even though the vendor never published the exact
    # sampling rate for this dataset.
    spectrum = np.abs(np.fft.rfft(centred))
    power = spectrum ** 2
    total_power = power.sum() + _EPS
    norm_freq = np.linspace(0.0, 1.0, power.size)

    spec_centroid = float((norm_freq * power).sum() / total_power)
    hf_ratio = float(power[norm_freq > 0.25].sum() / total_power)

    return np.array(
        [
            rms,
            x.max() - x.min(),
            kurtosis,
            skewness,
            crest_factor,
            shape_factor,
            impulse_factor,
            spec_centroid,
            hf_ratio,
        ],
        dtype=np.float64,
    )


def feature_column_names() -> list[str]:
    """Column order of the extracted feature matrix (36 vibration descriptors)."""
    return [f"{ch}_{d}" for ch in CHANNEL_NAMES for d in DESCRIPTORS]


def load_regime(load_pct: int) -> str:
    """Map a load percentage onto its coarse operating regime."""
    for low, high, label in LOAD_REGIMES:
        if low <= load_pct <= high:
            return label
    raise ValueError(f"Load {load_pct}% falls outside the defined regimes")


# --------------------------------------------------------------------------
# Recording-level processing
# --------------------------------------------------------------------------

def read_recording(path: Path) -> np.ndarray:
    """Load one raw ``*.txt`` recording into an (n_samples, 4) array."""
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=list(CHANNEL_NAMES),
        engine="python",
        skip_blank_lines=True,
    ).apply(pd.to_numeric, errors="coerce").dropna()
    return frame.to_numpy(dtype=np.float64)


def parse_recording_name(path: Path) -> tuple[str, int]:
    """``b30hz70.txt`` -> ("BrokenTooth", 70);  ``h30hz0.txt`` -> ("Healthy", 0)."""
    stem = path.stem.lower()
    condition = "BrokenTooth" if stem.startswith("b") else "Healthy"
    match = re.search(r"hz(\d+)$", stem)
    if match is None:
        raise ValueError(f"Cannot read load level from file name: {path.name}")
    return condition, int(match.group(1))


def segment_recording(samples: np.ndarray) -> np.ndarray:
    """Split an (n, 4) recording into non-overlapping windows.

    Non-overlapping windows are used deliberately: overlapping segments share
    raw samples, which would leak information between the training and test
    partitions and inflate every score in the comparison table.
    """
    usable = (samples.shape[0] // SEGMENT_LENGTH) * SEGMENT_LENGTH
    trimmed = samples[:usable]
    n_windows = usable // SEGMENT_LENGTH
    return trimmed.reshape(n_windows, SEGMENT_LENGTH, samples.shape[1])


def build_feature_frame(raw_root: str | Path, verbose: bool = True) -> pd.DataFrame:
    """Walk the raw data folders and build the full labelled feature table."""
    raw_root = Path(raw_root)
    # rglob rather than a fixed depth: the same recordings are distributed both
    # as two zips that unpack into "Healthy Data/" and "BrokenTooth Data/"
    # subfolders and as a flat folder, and both layouts should just work.
    recordings = sorted(raw_root.rglob("*.txt"))
    if not recordings:
        raise FileNotFoundError(
            f"No raw .txt recordings found under {raw_root.resolve()}. Download "
            "'Healthy Data.zip' and 'BrokenTooth Data.zip' from the source "
            "listed in README.md and unpack them there."
        )

    rows, meta = [], []
    for path in recordings:
        condition, load_pct = parse_recording_name(path)
        windows = segment_recording(read_recording(path))

        health = "Healthy" if condition == "Healthy" else "Broken"
        state = f"{health}-{load_regime(load_pct)}"

        for position, window in enumerate(windows):
            descriptors = np.concatenate(
                [channel_descriptors(window[:, c]) for c in range(window.shape[1])]
            )
            rows.append(descriptors)
            # `position_frac` records where the window sits inside its own
            # recording; it drives the chronological split and is NOT a feature.
            # `load_pct` is kept for analysis only -- never used as a predictor.
            meta.append((state, condition, load_pct,
                         position / max(len(windows) - 1, 1)))

        if verbose:
            print(f"  {path.name:<14} {state:<14} load={load_pct:>2}%  "
                  f"windows={len(windows)}")

    features = pd.DataFrame(rows, columns=feature_column_names())
    features[TARGET_COLUMN] = [m[0] for m in meta]
    features["health"] = [m[1] for m in meta]
    features["load_pct"] = [m[2] for m in meta]
    features["position_frac"] = [m[3] for m in meta]
    return features


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split so that the *last* 30 % of each recording becomes the test set.

    A plain random split would scatter neighbouring windows of the same
    recording across both partitions. Those windows are near-duplicates, so
    every model would score close to 100 % and the comparison table would say
    nothing. Holding out a contiguous later stretch of each recording is the
    honest analogue of "train on history, deploy on what comes next", which is
    how a condition monitoring model is actually used in the field.
    """
    cutoff = 1.0 - HOLDOUT_FRACTION
    keep = feature_column_names() + [TARGET_COLUMN]
    train = frame.loc[frame["position_frac"] < cutoff, keep]
    test = frame.loc[frame["position_frac"] >= cutoff, keep]
    return (
        train.sample(frac=1.0, random_state=17).reset_index(drop=True),
        test.sample(frac=1.0, random_state=17).reset_index(drop=True),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data", help="folder holding the raw recordings")
    parser.add_argument("--outdir", default=".", help="where to write the CSVs")
    args = parser.parse_args()

    print("Extracting condition-monitoring features from raw recordings...")
    table = build_feature_frame(args.raw)
    train_df, test_df = chronological_split(table)

    out = Path(args.outdir)
    train_df.to_csv(out / "train_data.csv", index=False)
    test_df.to_csv(out / "test_data.csv", index=False)

    print(f"\nTotal windows      : {len(table)}")
    print(f"Predictor columns  : {len(feature_column_names())}")
    print(f"Train / Test split : {len(train_df)} / {len(test_df)}")
    print(f"Class balance      :\n{table[TARGET_COLUMN].value_counts().to_string()}")
