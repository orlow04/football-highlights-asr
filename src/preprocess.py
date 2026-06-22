"""Etapa 0 — extrai e normaliza áudio para 16 kHz mono via ffmpeg.

Uso:
    python -m src.preprocess --input jogo_bruto.mp4 --output jogo.wav

Não aplica redução de ruído: o som de torcida é sinal útil para o SER (§3).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config  # noqa: E402


def extrai_audio(input_path: str, output_path: str, sr: int = 16000) -> str:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg não encontrado no PATH. Instale-o antes de rodar.")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16",
        output_path,
    ]
    print("→", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="vídeo/áudio bruto")
    ap.add_argument("--output", required=True, help="WAV de saída (16 kHz mono)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = extrai_audio(args.input, args.output, sr=cfg["audio"]["sample_rate"])
    print(f"OK: {out}")


if __name__ == "__main__":
    main()
