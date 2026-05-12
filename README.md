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
- [ ] **Common Voice dataset** — Or any other CV version
- [ ] **Tier-2 datasets** — Add CLARIN-NL datasets (N-Best 2008, CGN) with institutional access (https://opensource-spraakherkenning-nl.github.io/ASR_NL_results/)

---

## License

Code: Apache-2.0. 
Datasets vary: see individual dataset cards.
