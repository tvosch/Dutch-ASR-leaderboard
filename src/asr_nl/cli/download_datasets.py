#!/usr/bin/env python3
"""Download datasets for ASR NL leaderboard evaluation."""

import argparse
import logging
import os
from pathlib import Path

from datasets import load_dataset, load_from_disk

from asr_nl.datasets import DATASETS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_dataset(name: str, cfg: dict, save_to_disk: bool, data_dir: Path):
    print(f"\n{'='*60}")
    print(f"Dataset: {name} ({cfg['hf_id']}, config={cfg['config']})")
    print(f"License: {cfg.get('license', 'see dataset card')}")
    print(f"Notes: {cfg.get('notes', '—')}")
    print(f"{'='*60}")

    for split in cfg.get("splits", [cfg["split"]]):
        disk_path = data_dir / cfg["key"] / split
        if save_to_disk and disk_path.exists():
            logger.info(f"[{split}] Arrow snapshot already at {disk_path} — skipping.")
            continue
        
        logger.info(f"[{split}] Downloading...")
        try:
            ds = load_dataset(
                cfg["hf_id"],
                cfg["config"],
                split=split,
                trust_remote_code=cfg.get("trust_remote_code", False),
            )
        except Exception as e:
            logger.error(f"[{split}] FAILED: {e}")
            if "authentication" in str(e).lower() or "gated" in str(e).lower():
                logger.info(f"[{split}] → Run: huggingface-cli login")
            continue
        
        n = len(ds)
        logger.info(f"[{split}] {n} samples found — pre-loading ALL audio into cache...")
        audio_col = cfg["audio_col"]
        for i in range(n):
            _ = ds[i][audio_col]
            if (i + 1) % 200 == 0 or i == n - 1:
                logger.info(f"[{split}] ... {i + 1}/{n} audio files cached")
        logger.info(f"[{split}] All audio cached.")
        
        if save_to_disk:
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"[{split}] Saving Arrow snapshot to {disk_path}...")
            ds.save_to_disk(str(disk_path))
            logger.info(f"[{split}] Saved ({n} rows).")
        else:
            logger.info(f"[{split}] Done ({n} rows). Data in HF cache ($HF_HOME/datasets/).")


def verify_dataset(name: str, cfg: dict, data_dir: Path):
    logger.info(f"[verify] {name}")
    for split in cfg.get("splits", [cfg["split"]]):
        disk_path = data_dir / cfg["key"] / split
        if disk_path.exists():
            try:
                ds = load_from_disk(str(disk_path))
                logger.info(f"[{split}] Arrow snapshot OK — {len(ds)} rows at {disk_path}")
            except Exception as e:
                logger.error(f"[{split}] Arrow snapshot CORRUPT: {e}")
        else:
            try:
                ds = load_dataset(
                    cfg["hf_id"],
                    cfg["config"],
                    split=split,
                    trust_remote_code=cfg.get("trust_remote_code", False),
                )
                logger.info(f"[{split}] HF cache OK — {len(ds)} rows")
            except Exception as e:
                logger.error(f"[{split}] NOT CACHED: {e}")


def main():
    p = argparse.ArgumentParser(description="Download ASR NL leaderboard datasets")
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Which datasets to download (default: all).",
    )
    p.add_argument(
        "--save-to-disk",
        action="store_true",
        help="Save Arrow snapshots to --data-dir for instant reloads.",
    )
    p.add_argument(
        "--data-dir",
        default="data",
        help="Directory for Arrow snapshots (only used with --save-to-disk).",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Check download status without downloading.",
    )
    args = p.parse_args()
    
    hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface")
    logger.info(f"HF_HOME: {hf_home}")
    logger.info(f"Datasets: {args.datasets}")
    
    data_dir = Path(args.data_dir)
    
    for name in args.datasets:
        cfg = DATASETS[name]
        if args.verify_only:
            verify_dataset(name, cfg, data_dir)
        else:
            download_dataset(name, cfg, args.save_to_disk, data_dir)
    
    logger.info("\nDone.")
    if not args.verify_only and not args.save_to_disk:
        logger.info(
            "Tip: pass --save-to-disk to save Arrow snapshots for faster eval loading.\n"
            "Then set --data-dir in asr-nl-eval (or it auto-detects data/ if present)."
        )


if __name__ == "__main__":
    main()
