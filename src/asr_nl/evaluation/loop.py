"""Evaluation loop and metrics."""

import logging
import time
from pathlib import Path
from typing import Collection, Optional

import jiwer

from asr_nl.audio import to_mono
from asr_nl.backends import BaseBackend
from asr_nl.datasets import DATASETS, load_eval_dataset
from asr_nl.text import normalize_text

logger = logging.getLogger(__name__)


def result_normalizer_version(record: dict) -> int | None:
    """Normalizer version of a result file; files predating the field used v1."""
    if not record.get("normalize", True):
        return None
    return record.get("normalizer_version", 1)


def score_samples(
    per_sample: list[dict], language: str, normalize: bool = True, exclude: Collection[int] = ()
) -> dict:
    """
    (Re)compute scored text and corpus-level metrics from per-sample records.

    Mutates each successful record in place, setting reference_scored and
    hypothesis_scored. Failed samples (those with an "error") and samples with
    an empty scored reference are excluded from WER/CER, as are samples whose
    index is in `exclude` (known-bad dataset clips).
    """
    references, hypotheses, rtfs = [], [], []
    for s in per_sample:
        if "error" in s or s.get("index") in exclude:
            continue
        s["reference_scored"] = normalize_text(s["reference"], language) if normalize else s["reference"]
        s["hypothesis_scored"] = normalize_text(s["hypothesis"], language) if normalize else s["hypothesis"]
        if s["reference_scored"]:
            references.append(s["reference_scored"])
            hypotheses.append(s["hypothesis_scored"])
            rtfs.append(s["rtf"])

    if not references:
        return {}

    n_total = len(per_sample)
    n_failed = sum("error" in s for s in per_sample)
    return {
        "wer": round(jiwer.wer(references, hypotheses) * 100, 2),
        "cer": round(jiwer.cer(references, hypotheses) * 100, 2),
        "rtf": round(sum(rtfs) / len(rtfs), 4),
        "n_samples": len(references),
        "excluded": sorted(i for i in exclude if any(s.get("index") == i for s in per_sample)),
        "n_total": n_total,
        "n_failed": n_failed,
        "failure_rate_pct": round(n_failed / n_total * 100, 1),
    }


def evaluate_dataset(
    dataset_name: str,
    backend: BaseBackend,
    language: Optional[str] = None,
    max_samples: Optional[int] = None,
    data_dir: Optional[Path] = None,
    normalize: bool = True,
) -> dict:
    """
    Transcribe a dataset with `backend` and return metrics plus per-sample records.

    `language` defaults to the dataset's own language from DATASETS.
    """
    cfg = DATASETS[dataset_name]
    language = language or cfg["language"]
    logger.info(f"[{dataset_name}] {cfg['hf_id']} / {cfg['config']} language={language} normalize={normalize}")

    ds = load_eval_dataset(dataset_name, data_dir)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    per_sample = []
    for i, sample in enumerate(ds):
        # Every model expects mono; down-mix here so all backends get the same signal.
        audio = {"array": to_mono(sample[cfg["audio_col"]]["array"]),
                 "sampling_rate": sample[cfg["audio_col"]]["sampling_rate"]}
        record = {
            "index": i,
            "audio_duration": round(len(audio["array"]) / audio["sampling_rate"], 2),
            "reference": sample[cfg["text_col"]] or "",
        }
        t0 = time.perf_counter()
        try:
            hyp, rtf = backend.transcribe(audio, language)
        except Exception as e:
            logger.warning(f"[{dataset_name}] Sample {i} failed after {time.perf_counter() - t0:.1f}s: {e}")
            record.update(hypothesis="", error=str(e))
        else:
            record.update(hypothesis=hyp or "", rtf=round(rtf, 4), elapsed=round(time.perf_counter() - t0, 2))
            if i < 3:
                logger.info(f"[{dataset_name}] REF: {record['reference']}")
                logger.info(f"[{dataset_name}] HYP: {record['hypothesis']}")
        per_sample.append(record)

        if (i + 1) % 50 == 0:
            logger.info(f"[{dataset_name}] {i + 1}/{len(ds)} processed")

    metrics = score_samples(per_sample, language, normalize, exclude=cfg.get("exclude", {}))
    if not metrics:
        logger.warning(f"[{dataset_name}] No valid references - skipping")
        return {}
    if metrics["n_failed"]:
        logger.warning(f"[{dataset_name}] {metrics['n_failed']}/{metrics['n_total']} samples failed")
    logger.info(f"[{dataset_name}] WER={metrics['wer']}% CER={metrics['cer']}% RTF={metrics['rtf']}")

    return {
        **metrics,
        "dataset": {
            "hf_id": cfg["hf_id"],
            "config": cfg["config"],
            "split": cfg["split"],
            "revision": cfg.get("revision"),
            "language": language,
            "fingerprint": getattr(ds, "_fingerprint", None),
        },
        "per_sample": per_sample,
    }
