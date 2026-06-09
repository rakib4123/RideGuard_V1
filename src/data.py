"""Load + gate the two REAL uploaded datasets."""
from __future__ import annotations
import os, pathlib, yaml
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Resolve dataset locations across environments (sandbox, Google Colab, local).
# Override with env var RIDEGUARD_DATA_DIR, else search common locations.
_SEARCH_DIRS = [
    os.environ.get("RIDEGUARD_DATA_DIR", ""),
    "/mnt/user-data/uploads",
    "/content",                       # Google Colab default upload dir
    "/content/drive/MyDrive",
    str(ROOT / "data"),
    str(ROOT),
    ".",
]
_BIKE_NAMES = ["MotorBike_Accident.xlsx", "MotorBike_Accident_csv.xlsx"]
_DHAKA_NAMES = ["Accident_Dataset_Dhaka.xlsx", "Accident_Dataset_Dhaka_csv.xlsx"]


def _find(names) -> pathlib.Path:
    if isinstance(names, str):
        names = [names]
    for d in _SEARCH_DIRS:
        if not d:
            continue
        for name in names:
            p = pathlib.Path(d) / name
            if p.exists():
                return p
    raise FileNotFoundError(
        f"Could not find any of {names}. Put the file in one of "
        f"{[d for d in _SEARCH_DIRS if d]}, or set RIDEGUARD_DATA_DIR to its folder."
    )


BIKE_XLSX = None      # resolved lazily in load_* so missing files don't crash import
DHAKA_XLSX = None

isnum = pd.api.types.is_numeric_dtype
_CACHE = ROOT / "outputs"


def _read_cached(xlsx: pathlib.Path) -> pd.DataFrame:
    _CACHE.mkdir(exist_ok=True)
    pkl = _CACHE / (xlsx.stem + ".pkl")
    if pkl.exists() and pkl.stat().st_mtime >= xlsx.stat().st_mtime:
        return pd.read_pickle(pkl)
    df = pd.read_excel(xlsx)
    df.to_pickle(pkl)
    return df


def load_gate() -> dict:
    with open(ROOT / "config" / "feature_gate.yaml") as f:
        return yaml.safe_load(f)


def load_bike_raw() -> pd.DataFrame:
    """Full bike frame (used by the leakage receipt)."""
    return _read_cached(_find(_BIKE_NAMES))


def load_bike(gate: dict | None = None):
    """Returns X (gated pre-crash features), y (ordinal risk str), cat_cols."""
    gate = gate or load_gate()
    g = gate["tier1_behavioral"]
    df = _read_cached(_find(_BIKE_NAMES))
    df = df.loc[:, ~df.columns.str.contains("Unnamed|ENCODING", regex=True)]
    feats = [c for c in g["allowed"] if c in df.columns]
    X = df[feats].copy()
    # coerce the new pandas 'str' dtype -> object for downstream libs
    cat_cols = [c for c in feats if not isnum(X[c])]
    for c in cat_cols:
        X[c] = X[c].astype(object)
    y = df[g["target"]].astype(str)
    return X, y, cat_cols, g["target_order"]


def load_dhaka(gate: dict | None = None):
    """Returns the crash-only Dhaka frame (coords + intensity kept for the prior)."""
    gate = gate or load_gate()
    g = gate["tier2_spatial"]
    df = _read_cached(_find(_DHAKA_NAMES))
    return df, g



def forbidden_for(task: str, gate: dict | None = None) -> set[str]:
    gate = gate or load_gate()
    return set(gate[task].get("forbidden", []))
