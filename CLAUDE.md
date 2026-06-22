# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

This repository currently contains **only a design document** — `documentacao_projeto.md` (in Portuguese). No source code, dependencies, build system, or tests exist yet. The document is the authoritative specification and protocol; treat it as the source of truth when implementing. Most code below does not exist yet — it is the planned target described in the doc.

## Project

Academic audio/voice-processing assignment ("Atividade Final"). The goal is to detect **highlights** in football (soccer) broadcast narration by fusing two modalities:

1. **ASR** — transcribe narration with word/segment timestamps, then score a weighted domain lexicon ("gol", "pênalti", "na rede"…) per 1 s window, penalizing nearby negators ("quase", "anulado"…).
2. **SER / arousal** — estimate acoustic excitation (RMS energy, F0/pitch, spectral centroid), **not** categorical emotion. This framing is deliberate and should be preserved in any implementation and write-up.

The two per-1s curves are combined by **late fusion** (`s = α·s_kw + β·s_ser`), peaks are detected (`mean + k_sigma·std`, min distance, asymmetric cut window `pre=5s / pos=10s` because the reaction follows the play), and results are evaluated with WER/CER for ASR and precision/recall/F1 (with temporal tolerance τ) for detection.

## Key design decisions (don't silently override these)

- **Domain adaptation, not language transfer.** The base Parakeet v3 already supports Portuguese; the experiment measures adaptation to sports-narration jargon/prosody, not English→Portuguese failure.
- **Arousal estimation, not emotion classification.** Avoid IEMOCAP/RAVDESS-style discrete-emotion framing.
- **Two data roles:** a real PT-BR raw video is the main case (manual ground-truth annotation required); **SoccerNet-Echoes (EN)** is the fallback/prototyping set with ready event annotations — used to debug the pipeline, *not* to evaluate the PT-BR ASR.
- **Everything tunable is a hyperparameter** to be logged and ablated: the lexicon and its weights, the arousal weights (0.5/0.3/0.2), α/β, k_sigma, and the cut window. Planned to live in `configs/params.yaml`.

## Planned structure (from §10 of the doc)

```
src/preprocess.py  # stage 0: ffmpeg → 16 kHz mono WAV
src/asr.py         # stage 1: transcription + lexicon scoring
src/ser.py         # stage 2: arousal curve
src/fusion.py      # stage 3: late fusion on shared 1 s grid
src/detect.py      # stage 4: peak detection + segment merging
src/evaluate.py    # stage 5: WER/CER, P/R/F1
configs/params.yaml
notebooks/experimentos.ipynb
data/pt/  data/soccernet/
```

## Commands

Audio preprocessing (the one concrete command in the spec):

```bash
ffmpeg -i jogo_bruto.mp4 -vn -ac 1 -ar 16000 -sample_fmt s16 jogo.wav
```

**16 kHz mono is required by Parakeet.** Do not apply aggressive noise reduction — crowd noise is useful signal for the SER stage; log any filtering applied.

## Stack / dependencies (planned, not yet pinned)

- ASR: `alexandreacff/parakeet-tdt-0.6b-v3-ptBR-plus`. **Model card verified:** ships **`.nemo` only** (`parakeet-tdt-0.6b-v3-datasets-ptbr-e-podcasts.nemo`, 2.51 GB) → inference is **NeMo-only**. **Runtime env: RTX 4090 (24 GB) via SSH** — Ada Lovelace (compute 8.9) needs **PyTorch built for CUDA 12.x** (cu121), not 11.x. Install in an isolated conda env: `torch torchaudio` from the cu121 index, then `nemo_toolkit[asr]`. 24 GB is ample for 0.6B inference and long-audio local attention (chunking is for speed, not VRAM). Run long transcriptions under `tmux`/`nohup` (SSH). No `transformers`/`safetensors` path — don't add one. Trained on **PT-BR + podcasts (no sports narration)**, which is the foundation for the domain-gap argument in Experiment 1. **Still unverified at runtime:** word-level timestamps — TDT predicts token durations so it should support them, but confirm `hyp[0].timestamp["word"]` is populated; if only `segment` comes back, cuts fall to segment granularity. Whisper-PT is the documented plan-B (loses the E1 axis).
- SER: `librosa` for the acoustic-feature baseline; optional `audeering/wav2vec2-large-robust-...-msp-dim` as the advanced variant.
- Detection/eval: `scipy.signal.find_peaks`, `numpy`, `jiwer` (WER/CER).

## Experiments to support (§9)

- **E1 (ASR):** base v3 vs. PT-BR fine-tune vs. domain-lexicon residual error — WER, CER, and WER restricted to the domain lexicon.
- **E2 (fusion ablation):** ASR-only (β=0) vs. SER-only (α=0) vs. fusion, plus an α/β · k_sigma sweep; compare against a naïve RMS-only baseline. Report F1 across τ ∈ {5, 10, 15}s.

Reproducibility expectations: fix seeds, record library versions, version the lexicon and hyperparameters, and guard against train/test leakage in the fine-tune.
