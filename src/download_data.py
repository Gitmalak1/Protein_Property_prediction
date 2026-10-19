"""Download the public processed protein-solubility CSV."""
from pathlib import Path
import requests
from .data_utils import ROOT, load_config

def main():
    dest = ROOT / "data" / "raw_solubility.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Dataset already exists: {dest}")
        return
    url = load_config()["dataset_url"]
    print(f"Downloading dataset from: {url}")
    try:
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        if not response.content or b"<html" in response.content[:500].lower():
            raise RuntimeError("URL returned empty content or an HTML page, not a CSV.")
        dest.write_bytes(response.content)
    except Exception as exc:
        raise SystemExit(
            f"Dataset download failed: {exc}\\nDownload the processed CSV from "
            "https://github.com/xiaomizhou616/Protein_solubility_regression_model "
            "and save it as data/raw_solubility.csv."
        )
    print(f"Saved dataset to {dest} ({dest.stat().st_size:,} bytes).")
    print("Next: inspect the column names, set target_column in config.json if needed, then run python -m src.train")

if __name__ == "__main__":
    main()
