"""Dataset loading utilities."""

import logging
from pathlib import Path
from typing import Optional

from datasets import load_dataset, load_from_disk

from .configs import DATASETS

logger = logging.getLogger(__name__)


def load_eval_dataset(dataset_name: str, data_dir: Optional[Path]):
    """
    Load a dataset split, preferring an Arrow snapshot from data_dir.
    
    Falls back to load_dataset() which reads from HF cache or re-downloads.
    """
    cfg = DATASETS[dataset_name]
    
    if data_dir is not None:
        snapshot = data_dir / cfg["key"] / cfg["split"]
        if snapshot.exists():
            logger.info(f"[{dataset_name}] Loading Arrow snapshot from {snapshot}")
            check_snapshot_revision(dataset_name, snapshot)
            return load_from_disk(str(snapshot))
    
    logger.info(f"[{dataset_name}] Loading from HF cache / network ...")
    return load_dataset(
        cfg["hf_id"],
        cfg["config"],
        split=cfg["split"],
        revision=cfg.get("revision"),
        trust_remote_code=cfg.get("trust_remote_code", False),
    )


REVISION_FILE = "revision.txt"


def check_snapshot_revision(dataset_name: str, snapshot: Path) -> None:
    """Refuse a snapshot downloaded from a different revision than the pinned one."""
    expected = DATASETS[dataset_name].get("revision")
    marker = snapshot / REVISION_FILE
    if expected is None:
        return
    if not marker.exists():
        logger.warning(f"[{dataset_name}] Snapshot has no {REVISION_FILE}; cannot verify it is revision {expected[:8]}")
        return
    found = marker.read_text().strip()
    if found != expected:
        raise ValueError(
            f"[{dataset_name}] Snapshot {snapshot} is revision {found[:8]}, config pins {expected[:8]}. "
            "Re-download it with asr-nl-download --save-to-disk."
        )
