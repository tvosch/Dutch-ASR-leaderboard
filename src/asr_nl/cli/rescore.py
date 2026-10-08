"""Re-score result files with the current normalizer and clip exclusions, without re-running any model.

Works because every result file stores the raw reference and hypothesis per sample.
"""

import argparse
import json
from pathlib import Path

from asr_nl.datasets import DATASETS
from asr_nl.evaluation import result_normalizer_version, score_samples
from asr_nl.text import NORMALIZER_VERSION

LANGUAGE_BY_KEY = {cfg["key"]: cfg["language"] for cfg in DATASETS.values()}
EXCLUDE_BY_KEY = {cfg["key"]: cfg.get("exclude", {}) for cfg in DATASETS.values()}


def rescore(record: dict) -> dict[str, tuple[float, float]]:
    """Re-score `record` in place; return {dataset_key: (old_wer, new_wer)}."""
    changes = {}
    for key, result in record.get("results", {}).items():
        if not result.get("per_sample"):
            continue
        language = result.get("dataset", {}).get("language") or LANGUAGE_BY_KEY[key]
        metrics = score_samples(
            result["per_sample"], language, normalize=True, exclude=EXCLUDE_BY_KEY.get(key, {})
        )
        changes[key] = (result.get("wer"), metrics["wer"])
        result.update(metrics)
    record["normalize"] = True
    record["normalizer_version"] = NORMALIZER_VERSION
    return changes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path)
    out = p.add_mutually_exclusive_group(required=True)
    out.add_argument("--in-place", action="store_true", help="Overwrite the input files.")
    out.add_argument("--output-dir", type=Path, help="Write re-scored files here.")
    args = p.parse_args()

    for path in args.files:
        record = json.loads(path.read_text())
        old_version = result_normalizer_version(record)
        changes = rescore(record)
        print(f"{path.name}  (normalizer v{old_version} -> v{NORMALIZER_VERSION})")
        for key, (old, new) in changes.items():
            delta = f"{new - old:+.2f}" if old is not None else "n/a"
            print(f"    {key:15s} WER {old} -> {new}  ({delta})")

        target = path if args.in_place else args.output_dir / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
