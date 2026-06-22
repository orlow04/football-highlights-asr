"""Etapa 2 — estimador de excitação acústica (arousal), não emoção categórica (§5).

Uso:
    python -m src.ser --audio jogo.wav --out out/ser.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, zscore  # noqa: E402


def suaviza(x: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def curva_arousal(wav_path: str, cfg) -> np.ndarray:
    import librosa

    sr = cfg["audio"]["sample_rate"]
    janela = cfg["audio"]["window_s"]
    frame_s = cfg["audio"]["frame_s"]
    w = cfg["ser"]["weights"]

    y, sr = librosa.load(wav_path, sr=sr)
    hop = int(sr * frame_s)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    f0 = librosa.yin(y, fmin=cfg["ser"]["f0_min"], fmax=cfg["ser"]["f0_max"],
                     sr=sr, hop_length=hop)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]

    arousal = w["rms"] * zscore(rms) + w["f0"] * zscore(f0) + w["centroid"] * zscore(cent)

    # reamostra dos quadros (frame_s) para a grade de janela (média).
    fator = int(janela / frame_s)
    n = len(arousal) // fator
    arousal_g = arousal[:n * fator].reshape(n, fator).mean(axis=1)
    return suaviza(arousal_g, k=cfg["ser"]["smooth_k"])


def run(audio_path: str, config=None) -> dict:
    cfg = load_config(config)
    s_ser = curva_arousal(audio_path, cfg)
    return {"s_ser": s_ser.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    res = run(args.audio, config=args.config)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"OK: {args.out} | {len(res['s_ser'])} janelas")


if __name__ == "__main__":
    main()
