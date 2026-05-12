#!/usr/bin/env python3
"""Dutch ASR Leaderboard — evaluation script."""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import asr_nl

from asr_nl.backends import create_backend
from asr_nl.datasets import DATASETS
from asr_nl.evaluation import evaluate_dataset

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
        help="API key (or set OPENAI_API_KEY or RESEON8_API_KEY env var). Not stored in results.",
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
    c.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    
    e = p.add_argument_group("Evaluation")
    e.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
    )
    e.add_argument("--language", default="nl")
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
    
    return p.parse_args()


def main():
    args = parse_args()
    
    logger.info(f"Model: {args.model}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Normalize: {args.normalize}")
    
    backend = create_backend(args)
    data_dir = Path(args.data_dir) if args.data_dir else None
    
    try:
        results = {}
        for ds_name in args.datasets:
            key = DATASETS[ds_name]["key"]
            results[key] = evaluate_dataset(
                ds_name, backend, args.language,
                args.max_samples, data_dir, args.normalize
            )
    finally:
        backend.close()
    
    from asr_nl.backends import AudioAPIBackend, VLLMServerBackend, TransformersBackend, NeMoBackend

    def _backend_version(b) -> dict:
        try:
            if isinstance(b, VLLMServerBackend):
                import vllm
                mode = getattr(b._api, "api_mode", "transcriptions")
                return {"api_mode": mode, "vllm_version": vllm.__version__}
            if isinstance(b, TransformersBackend):
                import transformers
                return {"transformers_version": transformers.__version__}
            if isinstance(b, NeMoBackend):
                import nemo
                return {"nemo_version": nemo.__version__}
            if isinstance(b, AudioAPIBackend):
                mode = getattr(b, "api_mode", "transcriptions")
                if mode == "murmel":
                    import murmel
                    return {"api_mode": mode, "murmel_version": murmel.__version__}
                if mode == "reson8":
                    return {"api_mode": mode}
                if mode == "elevenlabs":
                    import elevenlabs
                    return {"api_mode": mode, "elevenlabs_version": elevenlabs.__version__}
                import vllm
                return {"api_mode": mode, "vllm_version": vllm.__version__}
        except Exception:
            pass
        return {}

    record = {
        "model_id": args.model,
        "model_name": args.model_name or args.model,
        "submission_date": str(date.today()),
        "license": args.license,
        "params_billions": args.params_billions,
        "backend_used": args.backend,
        "normalize": args.normalize,
        "run_info": {
            **_backend_version(backend),
            "package_version": asr_nl.__version__,
        },
        "results": results,
    }
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.model.replace("/", "__")
    out_path = output_dir / f"{safe_name}.json"
    
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        existing.setdefault("results", {}).update(record["results"])
        record["results"] = existing["results"]
    
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Result written to: {out_path}")
    logger.info("Commit this file to the leaderboard Space repo to publish results.")


if __name__ == "__main__":
    main()
