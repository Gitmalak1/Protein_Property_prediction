"""Predict protein solubility using a trained model bundle."""
import argparse
from pathlib import Path
import pandas as pd
import joblib
from .data_utils import ROOT, extract_sequence_features

def main():
    parser = argparse.ArgumentParser(description="Predict protein solubility from a sequence or feature CSV.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence", help="One amino-acid sequence (20 standard amino acids).")
    group.add_argument("--input", help="CSV file containing either a sequence column or model feature columns.")
    parser.add_argument("--output", default="reports/predictions.csv", help="Output CSV path.")
    args = parser.parse_args()
    model_path = ROOT / "models" / "protein_solubility_model.joblib"
    if not model_path.exists():
        raise SystemExit("Trained model not found. Run `python -m src.train` first.")
    bundle = joblib.load(model_path)
    if args.sequence:
        if not bundle["sequence_based"]:
            raise SystemExit("This model was trained on processed numeric features, not sequences. "
                             "Sequence prediction requires a sequence-based training dataset.")
        frame = pd.DataFrame([extract_sequence_features(args.sequence)])
    else:
        incoming = pd.read_csv(args.input)
        seq_col = next((c for c in incoming.columns if str(c).strip().lower() in
                        {"sequence", "seq", "aa_seq", "protein_sequence"}), None)
        if seq_col:
            if not bundle["sequence_based"]:
                raise SystemExit("Input contains sequences but saved model was trained on processed numeric features.")
            frame = pd.DataFrame([extract_sequence_features(s) for s in incoming[seq_col]])
        else:
            frame = incoming.reindex(columns=bundle["feature_columns"])
    missing = set(bundle["feature_columns"]) - set(frame.columns)
    if missing:
        raise SystemExit(f"Missing model features: {sorted(missing)}")
    pred = bundle["model"].predict(frame[bundle["feature_columns"]])
    result = pd.DataFrame({"predicted_solubility": pred})
    if args.sequence:
        result.insert(0, "sequence", args.sequence)
    elif "sequence" in incoming.columns:
        result.insert(0, "sequence", incoming["sequence"].values)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Saved predictions to {output}")

if __name__ == "__main__":
    main()
