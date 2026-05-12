"""Evaluation loop and metrics."""

import json
import logging
from pathlib import Path
from typing import Optional

import time

from asr_nl.datasets import DATASETS, load_eval_dataset
from asr_nl.text import normalize_text

from asr_nl.backends import BaseBackend

logger = logging.getLogger(__name__)


def evaluate_dataset(
    dataset_name: str,
    backend: BaseBackend,
    language: str = "nl",
    max_samples: Optional[int] = None,
    data_dir: Optional[Path] = None,
    normalize: bool = True,
) -> dict:
    """
    Evaluate a model on a dataset and return metrics.
    
    Returns dict with wer, cer, rtf, n_samples, n_failed, failure_rate_pct, per_sample.
    """
    from evaluate import load as load_metric
    
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")
    cfg = DATASETS[dataset_name]
    
    logger.info(f"[{dataset_name}] {cfg['hf_id']} / {cfg['config']} (normalize={'on' if normalize else 'OFF'})")
    
    ds = load_eval_dataset(dataset_name, data_dir)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    
    references, hypotheses, rtf_values = [], [], []
    per_sample = []
    n_failed = 0
    
    for i, sample in enumerate(ds):
        reference = sample[cfg["text_col"]] or ""
        audio_dur = len(sample[cfg["audio_col"]]["array"]) / sample[cfg["audio_col"]]["sampling_rate"]
        
        logger.info(f"[{dataset_name}] Sample {i+1}/{len(ds)} (audio: {audio_dur:.1f}s)")
        
        req_t0 = time.perf_counter()
        try:
            hyp, rtf = backend.transcribe(sample[cfg["audio_col"]], language)
        except Exception as e:
            elapsed = time.perf_counter() - req_t0
            n_failed += 1
            logger.warning(f"[{dataset_name}] Sample {i} failed after {elapsed:.1f}s: {e}")
            per_sample.append({
                "index": i,
                "audio_duration": round(audio_dur, 2),
                "reference": reference,
                "hypothesis": "",
                "error": str(e),
            })
            continue
        
        elapsed = time.perf_counter() - req_t0
        if i < 5 or elapsed > 10:
            hyp_short = (hyp or '')[:80]
            logger.info(f"[{dataset_name}] Sample {i+1} done in {elapsed:.1f}s (RTF={rtf:.3f}): {hyp_short}")
        
        ref_scored = normalize_text(reference, language) if normalize else reference
        hyp_scored = normalize_text(hyp, language) if normalize else hyp
        
        if i < 3 and ref_scored:
            logger.info(f"[{dataset_name}] REF: {ref_scored}")
            logger.info(f"[{dataset_name}] HYP: {hyp_scored}")
        
        per_sample.append({
            "index": i,
            "audio_duration": round(audio_dur, 2),
            "reference": reference,
            "reference_scored": ref_scored,
            "hypothesis": hyp,
            "hypothesis_scored": hyp_scored,
            "rtf": round(rtf, 4),
            "elapsed": round(elapsed, 2),
        })
        
        if ref_scored:
            references.append(ref_scored)
            hypotheses.append(hyp_scored)
            rtf_values.append(rtf)
        
        if (i + 1) % 50 == 0:
            logger.info(f"[{dataset_name}] {i + 1}/{len(ds)} processed")
    
    if not references:
        logger.warning(f"[{dataset_name}] No valid references - skipping")
        return {}
    
    wer = round(wer_metric.compute(predictions=hypotheses, references=references) * 100, 2)
    cer = round(cer_metric.compute(predictions=hypotheses, references=references) * 100, 2)
    rtf_mean = round(sum(rtf_values) / len(rtf_values), 4)
    n_total = len(ds)
    failure_rate = round(n_failed / n_total * 100, 1) if n_total > 0 else 0.0
    
    if n_failed > 0:
        logger.warning(f"[{dataset_name}] {n_failed}/{n_total} samples failed ({failure_rate}%)")
    
    logger.info(f"[{dataset_name}] WER={wer}% CER={cer}% RTF={rtf_mean} failures={failure_rate}%")
    
    return {
        "wer": wer,
        "cer": cer,
        "rtf": rtf_mean,
        "n_samples": len(references),
        "n_failed": n_failed,
        "failure_rate_pct": failure_rate,
        "per_sample": per_sample,
    }
