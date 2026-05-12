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
            return load_from_disk(str(snapshot))
    
    logger.info(f"[{dataset_name}] Loading from HF cache / network ...")
    return load_dataset(
        cfg["hf_id"],
        cfg["config"],
        split=cfg["split"],
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
