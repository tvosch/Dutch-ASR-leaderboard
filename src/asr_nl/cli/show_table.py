"""Print a leaderboard table of result files to stdout."""

import argparse
import json
from pathlib import Path

import pandas as pd

DATASET_LABELS = {"fleurs_nl": "FLEURS", "voxpopuli_nl": "VoxPopuli", "mls_nl": "MLS"}


def load_results(results_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]


def build_dataframe(records: list[dict], metric: str = "wer") -> pd.DataFrame:
    rows = []
    for r in records:
        row = {"Model": r.get("model_name", r.get("model_id", "?")), "Norm": r.get("normalizer_version", 1)}
        for key, label in DATASET_LABELS.items():
            row[label] = r.get("results", {}).get(key, {}).get(metric)
        rows.append(row)
    df = pd.DataFrame(rows)
    labels = list(DATASET_LABELS.values())
    # Average only over models that have every dataset, so averages are comparable.
    df["Average"] = df[labels].mean(axis=1, skipna=False).round(2)
    return df.sort_values(["Average", "Model"], na_position="last").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--metric", choices=["wer", "cer", "rtf"], default="wer")
    args = p.parse_args()

    records = load_results(args.results_dir)
    if not records:
        print(f"No result files found in {args.results_dir}/")
        return
    print(build_dataframe(records, args.metric).to_string(index=False, na_rep="—"))


if __name__ == "__main__":
    main()
