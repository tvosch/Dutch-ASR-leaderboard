"""Print a leaderboard table for Dutch ASR results to stdout."""

import argparse
from pathlib import Path

from ..gradio.app import build_dataframe, load_results, DATASET_KEYS, DATASET_LABELS


def print_table(results_dir: str = "results", sort_by: str = "fleurs_nl", leaderboard: bool = False):
    records = load_results()
    if not records:
        print(f"No result files found in {results_dir}/")
        return

    df = build_dataframe(records, primary_dataset=sort_by)

    if leaderboard:
        keep = ["Model"] + [c for c in df.columns if c in ("FLEURS WER", "VoxPopuli WER")]
        df = df[keep]
    else:
        dutch_labels = {DATASET_LABELS[k] for k in DATASET_KEYS}
        keep = [c for c in df.columns if any(c.startswith(label) for label in dutch_labels)
                or c in ("Model", "Type", "Params (M)", "License", "Submitted")]
        df = df[keep]

    print(df.to_string(index=False))
    print(f"\n{len(records)} model(s) loaded from results/")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sort-by",
        choices=list(DATASET_KEYS),
        default="fleurs_nl",
        help="Dataset to sort by WER (default: fleurs_nl)",
    )
    p.add_argument("--results-dir", default="results")
    p.add_argument(
        "--leaderboard",
        action="store_true",
        help="Print only model name with FLEURS and VoxPopuli WER",
    )
    args = p.parse_args()
    print_table(args.results_dir, args.sort_by, args.leaderboard)


if __name__ == "__main__":
    main()
