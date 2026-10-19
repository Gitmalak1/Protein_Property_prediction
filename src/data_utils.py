"""Data loading, validation, and sequence feature extraction."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AA = list("ACDEFGHIKLMNPQRSTVWY")
HYDROPATHY = dict(zip(AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                            -1.6, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, -0.9, -1.3, 4.2]))
TARGET_HINTS = ("solubility", "soluble", "sol", "label", "target", "y")

def load_config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

def extract_sequence_features(sequence):
    """Convert a protein sequence to transparent, composition-based numeric features."""
    seq = re.sub(r"\s+", "", str(sequence).upper())
    if not seq or any(aa not in AA for aa in seq):
        raise ValueError("Sequence must be non-empty and contain only the 20 standard amino acids.")
    n = len(seq)
    counts = {aa: seq.count(aa) / n for aa in AA}
    counts["sequence_length"] = n
    counts["mean_hydropathy"] = float(np.mean([HYDROPATHY[aa] for aa in seq]))
    counts["charged_fraction"] = sum(seq.count(a) for a in "DEKR") / n
    counts["aromatic_fraction"] = sum(seq.count(a) for a in "FWY") / n
    counts["glycine_fraction"] = seq.count("G") / n
    counts["proline_fraction"] = seq.count("P") / n
    return counts

def _target_column(df, configured=None):
    if configured and configured in df.columns:
        return configured
    lower = {str(c).strip().lower(): c for c in df.columns}
    for name in TARGET_HINTS:
        for key, original in lower.items():
            if name == key or name in key:
                if pd.api.types.is_numeric_dtype(df[original]):
                    return original
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        raise ValueError("No numeric target column found. Set target_column in config.json.")
    # Avoid silently selecting a target where possible; explicit config is preferable.
    raise ValueError("Could not confidently identify target. Set target_column in config.json. "
                     f"Numeric columns: {numeric}")

def load_training_data(path=None, target_column=None):
    path = Path(path) if path else ROOT / "data" / "raw_solubility.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run: python -m src.download_data")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty.")
    target = _target_column(df, target_column if target_column is not None else load_config().get("target_column"))
    df = df.dropna(subset=[target]).drop_duplicates().copy()
    y = pd.to_numeric(df[target], errors="coerce")
    keep = y.notna()
    df, y = df.loc[keep].copy(), y.loc[keep].astype(float)
    seq_col = next((c for c in df.columns if str(c).strip().lower() in {"sequence", "seq", "aa_seq", "protein_sequence"}), None)
    if seq_col:
        X = pd.DataFrame([extract_sequence_features(s) for s in df[seq_col]], index=df.index)
    else:
        excluded = {target}
        for c in df.columns:
            name = str(c).strip().lower()
            if name in {"id", "name", "protein", "protein_id", "uniprot", "accession"}:
                excluded.add(c)
        X = df.drop(columns=list(excluded), errors="ignore").select_dtypes(include=[np.number]).copy()
        # Remove columns that are entirely missing or constant.
        X = X.dropna(axis=1, how="all")
        X = X.loc[:, X.nunique(dropna=True) > 1]
    if X.empty:
        raise ValueError("No usable numeric features found after removing target/IDs.")
    X = X.replace([np.inf, -np.inf], np.nan)
    aa_feature_count = sum(1 for c in X.columns if str(c).strip().upper() in AA)
    sequence_compatible = bool(seq_col) or aa_feature_count >= 15
    summary = {"rows_after_cleaning": int(len(X)), "feature_count": int(X.shape[1]),
               "target_column": str(target), "feature_columns": [str(c) for c in X.columns],
               "sequence_based": sequence_compatible, "sequence_column_present": bool(seq_col),
               "amino_acid_composition_features": int(aa_feature_count),
               "target_min": float(y.min()), "target_max": float(y.max())}
    return X, y.loc[X.index], summary
