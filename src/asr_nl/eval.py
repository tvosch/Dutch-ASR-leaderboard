#!/usr/bin/env python3
"""Dutch ASR Leaderboard — evaluation script."""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import asr_nl
from asr_nl.backends import create_backend
from asr_nl.datasets import DATASETS
from asr_nl.evaluation import evaluate_dataset, result_normalizer_version
from asr_nl.text import NORMALIZER_VERSION

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Dutch ASR Leaderboard — evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True, help="HuggingFace model ID or API model name.")

    g = p.add_argument_group("Backend")
    g.add_argument(
        "--backend",
        choices=["vllm", "api", "transformers", "nemo"],
        default="vllm",
        help=(
            "vllm: spawn a local vLLM server (default). "
            "api: use an OpenAI-compatible HTTP endpoint. "
            "transformers: HuggingFace pipeline. "
            "nemo: NVIDIA NeMo ASR."
        ),
    )
    g.add_argument(
        "--api-base-url",
        default=None,
        help="OpenAI-compatible base URL (required for --backend api).",
    )
    g.add_argument(
        "--api-key",
        default=None,
        help="API key. Prefer --api-key-env so keys stay out of job files and shell history.",
    )
    g.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable to read the API key from when --api-key is not given (default: OPENAI_API_KEY).",
    )
    g.add_argument(
        "--api-mode",
        choices=["transcriptions", "chat", "reson8", "murmel", "elevenlabs"],
        default="transcriptions",
        help=(
            "transcriptions: POST /v1/audio/transcriptions (Whisper-style). "
            "chat: POST /v1/chat/completions with audio content (multimodal models). "
            "reson8: POST /v1/speech-to-text/prerecorded (Reson8 API). "
            "murmel: Use Murmel API client (requires murmel-python package). "
            "elevenlabs: Use ElevenLabs Scribe API (requires elevenlabs package)."
        ),
    )

    v = p.add_argument_group("vLLM options")
    v.add_argument("--vllm-port", type=int, default=8080)
    v.add_argument("--tensor-parallel-size", type=int, default=1)
    v.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    v.add_argument(
        "--vllm-args",
        default="",
        help=(
            "Extra flags passed verbatim to 'vllm serve', as a single quoted string. "
            "Example: --vllm-args \"--compilation_config '{\\\"cudagraph_mode\\\": \\\"PIECEWISE\\\"}'\""
        ),
    )

    c = p.add_argument_group("Compute")
    c.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available).")

    e = p.add_argument_group("Evaluation")
    e.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
    )
    e.add_argument("--language", default=None, help="Override the per-dataset language (debug).")
    e.add_argument("--max-samples", type=int, default=None, help="Cap per dataset (debug).")
    e.add_argument(
        "--data-dir",
        default=None,
        help="Directory with Arrow snapshots from --save-to-disk. If not set, loads from HF cache.",
    )
    e.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply text normalization before scoring (default: on).",
    )

    m = p.add_argument_group("Metadata")
    m.add_argument("--model-name", default=None)
    m.add_argument("--license", default="unknown")
    m.add_argument("--params-billions", type=float, default=None, help="Model size in billions (e.g., 1.7 for 1.7B).")
    m.add_argument("--output-dir", default="results")

    args = p.parse_args()
    if args.api_key is None:
        args.api_key = os.environ.get(args.api_key_env)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def main():
    args = parse_args()

    logger.info(f"Model: {args.model}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Normalize: {args.normalize}")

    output_dir = Path(args.output_dir)
    out_path = output_dir / f"{args.model.replace('/', '__')}.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else None
    if existing and result_normalizer_version(existing) != (NORMALIZER_VERSION if args.normalize else None):
        sys.exit(
            f"{out_path} was scored with normalizer_version={existing.get('normalizer_version')}; "
            f"run `asr-nl-rescore {out_path} --in-place` first so merged results stay comparable."
        )

    backend = create_backend(args)
    data_dir = Path(args.data_dir) if args.data_dir else None

    try:
        run_info = backend.run_info()
        results = {}
        for ds_name in args.datasets:
            key = DATASETS[ds_name]["key"]
            results[key] = evaluate_dataset(
                ds_name, backend, args.language,
                args.max_samples, data_dir, args.normalize
            )
    finally:
        backend.close()

    record = {
        "model_id": args.model,
        "model_name": args.model_name or args.model,
        "submission_date": str(date.today()),
        "license": args.license,
        "params_billions": args.params_billions,
        "backend_used": args.backend,
        "normalize": args.normalize,
        "normalizer_version": NORMALIZER_VERSION if args.normalize else None,
        "run_info": {
            **run_info,
            "package_version": asr_nl.__version__,
            "git_commit": _git_commit(),
            "container": os.environ.get("APPTAINER_CONTAINER"),
            "container_lock_sha": os.environ.get("ASR_NL_LOCK_SHA"),
        },
        "results": results,
    }

    if existing:
        record["results"] = {**existing.get("results", {}), **record["results"]}

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    logger.info(f"Result written to: {out_path}")
    logger.info("Commit this file to the leaderboard Space repo to publish results.")


if __name__ == "__main__":
    main()
