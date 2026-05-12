"""OpenAI-compatible API backends (AudioAPIBackend and VLLMServerBackend)."""

import base64
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

from asr_nl.audio import audio_to_wav_bytes, find_free_port, resample_to_16k
from asr_nl.audio.processing import TARGET_SR

logger = logging.getLogger(__name__)


class AudioAPIBackend:
    """
    Unified OpenAI-compatible audio client.
    
    Works with vLLM servers, OpenAI API, or any OpenAI-compatible ASR endpoint.
    """
    
    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str,
        api_mode: str = "transcriptions",
    ):
        from openai import OpenAI
        
        self.model_id = model_id
        self.api_mode = api_mode
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "dummy",
            timeout=120.0,
        )
        logger.info(f"Initialized AudioAPIBackend for {model_id} at {base_url}")
    
    def transcribe(self, audio: dict, language: str = "nl") -> tuple[str, float]:
        """Transcribe audio via API."""
        array = audio["array"]
        sr = audio["sampling_rate"]
        duration = len(array) / sr

        array = resample_to_16k(array, sr)
        wav_bytes = audio_to_wav_bytes(array, TARGET_SR)

        t0 = time.perf_counter()

        if self.api_mode == "chat":
            text = self._via_chat(wav_bytes, language)
        elif self.api_mode == "reson8":
            text = self._via_reson8(wav_bytes, language)
        elif self.api_mode == "murmel":
            text = self._via_murmel(wav_bytes, language)
        elif self.api_mode == "elevenlabs":
            text = self._via_elevenlabs(wav_bytes, language)
        else:
            text = self._via_transcriptions(wav_bytes, language)

        rtf = (time.perf_counter() - t0) / duration if duration > 0 else 0.0
        return text, rtf
    
    def _via_transcriptions(self, wav_bytes: bytes, language: str) -> str:
        """POST /v1/audio/transcriptions (Whisper-style)."""
        import requests as req_lib
        
        url = str(self.client.base_url).rstrip("/") + "/audio/transcriptions"
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"model": self.model_id, "max_tokens": 2048}
        if language:
            data["language"] = language
        headers = {"Authorization": f"Bearer {self.client.api_key}"}
        
        try:
            resp = req_lib.post(url, files=files, data=data, headers=headers, timeout=120)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"Transcriptions API failed, retrying without language: {e}")
            data.pop("language", None)
            resp = req_lib.post(url, files=files, data=data, headers=headers, timeout=120)
            resp.raise_for_status()
        
        parsed = resp.json()
        return parsed.get("text", "") or ""
    
    def close(self):
        """Close the client (no-op for API backend)."""
        pass

    def _via_chat(self, wav_bytes: bytes, language: str) -> str:
        """POST /v1/chat/completions with audio content (multimodal models)."""
        import re
        
        b64 = base64.b64encode(wav_bytes).decode()
        lang_token = f"<|{language}|>" if language else "<|en|>"
        
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": lang_token},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": b64, "format": "wav"},
                        }
                    ],
                }
            ],
            max_tokens=2048,
        )
        
        # Extract transcription from structured response
        text = resp.choices[0].message.content or ""
        m = re.search(r"<asr_text>(.*?)(?:</asr_text>|$)", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text
    
    def _via_reson8(self, wav_bytes: bytes, language: str) -> str:
        """POST /v1/speech-to-text/prerecorded (Reson8 API)."""
        import requests as req_lib
        
        url = str(self.client.base_url).rstrip("/") + "/speech-to-text/prerecorded"
        headers = {
            "Authorization": f"ApiKey {self.client.api_key}",
            "Content-Type": "application/octet-stream",
        }
        resp = req_lib.post(url, data=wav_bytes, headers=headers, timeout=120)
        resp.raise_for_status()
        parsed = resp.json()
        return parsed.get("text") or ""

    def _via_elevenlabs(self, wav_bytes: bytes, language: str) -> str:
        """POST /v1/speech-to-text (ElevenLabs Scribe API)."""
        from elevenlabs.client import ElevenLabs

        el_client = ElevenLabs(api_key=self.client.api_key)
        response = el_client.speech_to_text.with_raw_response.convert(
            file=("audio.wav", wav_bytes, "audio/wav"),
            model_id=self.model_id,
            language_code=language if language else None,
        )
        char_cost = response.headers.get("x-character-count")
        if char_cost:
            logger.info(f"ElevenLabs character cost: {char_cost}")
        return response.data.text or ""

    def _via_murmel(self, wav_bytes: bytes, language: str) -> str:
        """Use Murmel API client (requires murmel-python package)."""
        import tempfile
        from pathlib import Path

        from murmel import Murmel

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            temp_path = Path(f.name)

        try:
            client = Murmel(
                api_key=self.client.api_key or "dummy",
                base_url=str(self.client.base_url) if str(self.client.base_url) != "dummy" else None,
            )
            result = client.transcribe(temp_path, language=language)
            return result.text
        finally:
            temp_path.unlink()


class VLLMServerBackend:
    """
    Spawns a local vLLM server and delegates to AudioAPIBackend.
    """
    
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        port: Optional[int] = None,
        api_mode: str = "transcriptions",
        startup_timeout: int = 180,
        extra_args: str = "",
    ):
        self._process: Optional[subprocess.Popen] = None
        self.port = port or find_free_port()

        # Patch model config if needed
        effective_model_id = self._maybe_patch_config(model_id)

        # Spawn vLLM server
        cmd = [
            "vllm", "serve", effective_model_id,
            "--port", str(self.port),
            "--dtype", dtype,
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--gpu-memory-utilization", "0.8",
            "--trust-remote-code",
        ]
        if device == "cpu":
            cmd += ["--device", "cpu"]
        if extra_args:
            import shlex
            cmd += shlex.split(extra_args)
        
        logger.info(f"Spawning vLLM server: {' '.join(cmd)}")
        self._log_file = f"vllm_server_{self.port}.log"
        logger.info(f"vLLM server logs: {self._log_file}")
        
        with open(self._log_file, "w") as flog:
            self._process = subprocess.Popen(
                cmd,
                stdout=flog,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
        
        self._wait_for_healthy(startup_timeout)
        self._api = AudioAPIBackend(
            model_id, f"http://localhost:{self.port}/v1", "dummy", api_mode
        )
    
    def _maybe_patch_config(self, model_id: str) -> str:
        """Patch decoder_start_token_id if missing (e.g., Cohere ASR)."""
        try:
            from transformers import AutoConfig, AutoTokenizer
            from huggingface_hub import hf_hub_download
            
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            if getattr(config, "decoder_start_token_id", None) is None:
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                start_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
                if start_id is not None and start_id != -1:
                    cached_config = hf_hub_download(model_id, "config.json")
                    import json
                    with open(cached_config) as f:
                        cfg_data = json.load(f)
                    cfg_data["decoder_start_token_id"] = start_id
                    with open(cached_config, "w") as f:
                        json.dump(cfg_data, f, indent=2)
                    logger.info(f"Patched config.json at {cached_config}")
        except Exception as e:
            logger.warning(f"Failed to patch config: {e}")
        
        return model_id
    
    def _wait_for_healthy(self, timeout: int):
        """Wait for vLLM server to become healthy."""
        import urllib.request
        
        url = f"http://localhost:{self.port}/health"
        deadline = time.time() + timeout
        
        logger.info(f"Waiting for vLLM server on :{self.port} (up to {timeout}s)...")
        
        while time.time() < deadline:
            if self._process.poll() is not None:
                try:
                    out = Path(self._log_file).read_text(errors="replace")[-3000:]
                except Exception:
                    out = "(log unavailable)"
                raise RuntimeError(f"vLLM process exited early:\n{out}")
            
            try:
                urllib.request.urlopen(url, timeout=2)
                logger.info("vLLM server is ready.")
                return
            except Exception:
                time.sleep(2)
        
        self.close()
        raise RuntimeError(
            f"vLLM server did not become healthy within {timeout}s. "
            "Model may not be supported."
        )
    
    def transcribe(self, audio: dict, language: str = "nl") -> tuple[str, float]:
        """Delegate to AudioAPIBackend."""
        return self._api.transcribe(audio, language)
    
    def close(self):
        """Shutdown vLLM server."""
        if self._process is not None:
            logger.info("Shutting down vLLM server...")
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except Exception as e:
                logger.warning(f"Failed to kill vLLM process: {e}")
            self._process = None
