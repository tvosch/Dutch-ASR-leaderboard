# Dutch ASR Leaderboard

Community leaderboard for Dutch Automatic Speech Recognition models.

**Live leaderboard:** https://huggingface.co/spaces/tvosch/Dutch-ASR-Leaderboard

---

## Benchmark Datasets

| Dataset | Type | Description |
|---------|------|-------------|
| **FLEURS** | Read speech | Parallel sentences translated from Wikipedia. Studio-quality. |
| **VoxPopuli** | Spontaneous | European Parliament speech. Formal, domain-adaptation test. |
| **Multilingual LibriSpeech** | Read speech | Audiobook style. |
| **Common Voice 25** | Read speech | Crowd-sourced volunteer recordings. Wide speaker/accent/device variety. Gated on HF Hub (click-through); also available ungated via the Mozilla Data Collective. |

---

## Quick Start

```bash
git clone https://github.com/tvosch/Dutch-ASR-Leaderboard
cd Dutch-ASR-Leaderboard
pip install -e .

python -m asr_nl.eval \
    --model "openai/whisper-large-v3" \
    --backend vllm \
```

Results are written to `results/<model>.json`. Add them to the leaderboard via PR.

---

## Roadmap

- [ ] **Dutch text normalization**: Dutch-specific handling (https://github.com/ThomasKluiters/european-normalizer)
- [x] **Common Voice dataset** — CV 25 NL config available as `common_voice_25` (leaderboard Space column pending). Note: most Dutch-specific models fine-tune on CV train; same-version splits are speaker-disjoint, but flag this like the existing benchmaxxing note.
- [ ] **Tier-2 datasets** — Add CLARIN-NL datasets (N-Best 2008, CGN) with institutional access (https://opensource-spraakherkenning-nl.github.io/ASR_NL_results/)

---

## License

Code: Apache-2.0. 
Datasets vary: see individual dataset cards.
